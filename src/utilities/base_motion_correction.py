import os 
import ants
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import subprocess
from src.utilities.utils import load_nifti_data


class MotionCorrection:
    def __init__(self,
                 reference_volume=None,
                 registration_method='SimpleITK',
                 mask=None,
                 verbose=False,
                 interleave_factor=3,
                 hierarchical=True):
        self.reference_volume = reference_volume
        self.registration_method = registration_method
        self.mask = mask
        self.verbose = verbose
        self.interleave_factor = interleave_factor
        self.hierarchical = hierarchical
        self._volumes = []
        self._transformations = {}
        
    def _get_filenames(self, filepath):
        path, filename = os.path.split(filepath)
        base, _ = os.path.splitext(filename)
        base, _ = os.path.splitext(base)
        return path, base
    
    def _create_reference(self, bold_nii, mask_nii, ref_nii=None):
        bold_img = nib.load(bold_nii)
        img = load_nifti_data(bold_nii)
        mask = load_nifti_data(mask_nii)
        
        if mask.shape != img.shape[:3]:
            raise ValueError("Mask dimensions do not match the input image dimensions.")
        
        n_x, n_y, n_z, n_t = img.shape
        mse = np.zeros(n_t)
        
        for i in range(n_t):
            f1 = img[..., i]
            mse_sum = 0
            count = 0
            for j in range(n_t):
                if i != j:
                    f2 = img[...,j]
                    diff = (f1-f2)**2
                    mse_sum += np.sum(diff[mask > 0])
            
            mse[i] = mse_sum / ((n_t - 1) * np.sum(mask))
                
        # Threshold based on 10th percentile? of mse
        thr = np.quantile(mse, 0.1)
        valid_indices = np.where(mse <= thr)[0]
        ref_img = img[..., valid_indices]
        ref_img = np.mean(ref_img, axis=3)
        ref_header = bold_img.header.copy()
        ref_header['dim'][0] = 3 
        ref_header['dim'][4] = 1  
        reference = nib.Nifti1Image(ref_img, bold_img.affine, ref_header)
        if ref_nii:
            nib.save(reference, ref_nii)
        else:
            path, base = self._get_filenames(bold_nii)
            nib.save(reference, os.path.join(path, f"{base}_reference.nii.gz"))
        print("Reference image generated and saved.")
        
    def _initialize_registration(self, fixed_image, moving_image, mask, registration_type):
        if registration_type == 'SimpleITK':
            return SimpleItkRegistration(
                fixed=fixed_image,
                moving=moving_image,
                use_fixed_mask=True,
                use_moving_mask=True,
                registration_type="Affine",
                interpolator="Linear",
                metric="Correlation",
                metric_params=None,
                optimizer="RegularStepGradientDescent",
                optimizer_params={
                    "minStep": 1e-6,
                    "numberOfIterations": 200,
                    "gradientMagnitudeTolerance": 1e-6,
                    "learningRate": 1
                },
                scales_estimator="PhysicalShift",
                use_multiresolution_framework=True,
                shrink_factors=[2, 1],
                smoothing_sigmas=[1, 0],
                use_verbose=self.verbose
            )
        elif registration_type == 'RegAladin':
            return RegAladin(
                fixed=fixed_image,
                moving=moving_image,
                use_fixed_mask=True,
                use_moving_mask=True,
                use_verbose=self.verbose,
                options="-voff",
                registration_type="Rigid"
            )
        elif registration_type == 'antsRegistration':
            return ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                mask=mask,
                type_of_transform="SyNOnly",
                regIterations=(100, 75, 20, 0),
                verbose=self.verbose
            )
        elif registration_type == 'flirt':
            # Add FLIRT registration setup here
            pass
        else:
            raise ValueError("Unsupported registration method.")
        
    def _save_transformations(self, transformations, suffix):
        for name, transform in transformations.items():
            path, base = self._get_filenames(name)
            transform_path = os.path.join(path, f"{base}_{suffix}_transform.txt")
            sitk.WriteTransform(transform, transform_path)
            print(f"Transformation saved to {transform_path}")
            
    def _split_slices_based_on_interleave(self, slices):
        sub_volumes = []
        num_slices = len(slices)
        for i in range(0, num_slices, self.interleave):
            sub_volumes.append(slices[i:i+self.interleave])
        return sub_volumes
    
    def _run_registration(self, fixed_image, moving_image, method, mask=None):
        reg = self._initialize_registration(fixed_image, moving_image, mask, self.registration_method)
        reg.set_fixed(fixed_image)
        reg.set_moving(moving_image)
        reg.run()
        return reg.get_warped_moving_sitk(), reg.get_registration_transform_sitk()  
    
    def _volume_to_volume_registration(self):
        v2v_transforms = {}
        for i, volume in enumerate(self._volumes):
            txt = f"Volume-to-Volume Registration -- Volume {i + 1}/{len(self._volumes)}"
            if self.verbose:
                print(txt)

            warped_volume, transform_sitk = self._run_registration(self.reference_volume, volume, self.registration_method, self.mask)
            v2v_transforms[volume.get_filename()] = transform_sitk

        self._save_transformations(v2v_transforms, 'v2v')

    def _slice_set_to_volume_registration(self):
        ss2v_transforms = {}
        for i, volume in enumerate(self._volumes):
            slices = volume.get_slices()
            sub_volumes = self._split_slices_based_on_interleave(slices)

            for interleave_set, sub_volume in enumerate(sub_volumes):
                txt = f"Slice Set-to-Volume Registration -- Volume {i + 1}/{len(self._volumes)} -- Interleave Set {interleave_set + 1}/{len(sub_volumes)}"
                if self.verbose:
                    print(txt)

                warped_volume, transform_sitk = self._run_registration(self.reference_volume, sub_volume, self.registration_method, self.mask)
                ss2v_transforms[volume.get_filename()] = transform_sitk

        self._save_transformations(ss2v_transforms, 'ss2v')

    def _slice_to_volume_registration(self):
        s2v_transforms = {}
        for i, volume in enumerate(self._volumes):
            slices = volume.get_slices()

            for slice in slices:
                txt = f"Slice-to-Volume Registration -- Volume {i + 1}/{len(self._volumes)} -- Slice {slice.get_slice_number()}/{len(slices)}"
                if self.verbose:
                    print(txt)

                warped_slice, transform_sitk = self._run_registration(self.reference_volume, slice, self.registration_method, self.mask)
                s2v_transforms[slice.get_slice_number()] = transform_sitk

        self._save_transformations(s2v_transforms, 's2v')

    def run(self):
        if self.hierarchical:
            print("Starting hierarchical registration...")
            self._volume_to_volume_registration()  # Step 1: Volume-to-Volume
            self._slice_set_to_volume_registration()  # Step 2: Slice Set-to-Volume
            self._slice_to_volume_registration()  # Step 3: Slice-to-Volume
        else:
            print("Starting volume-to-volume registration...")
            self._volume_to_volume_registration()

        print("Registration completed.")