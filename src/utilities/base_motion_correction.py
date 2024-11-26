import os 
import ants
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import subprocess
from src.utilities.utils import load_nifti_data , get_filenames
from src.utilities.external_running import ExternalToolRunner


class MotionCorrection:
    def __init__(self,
                 reference_volume=None,
                 motion_corrected_nii=None,
                 registration_method='SimpleITK',
                 registration_type='rigid',
                 mask=None,
                 verbose=True,
                 interleave_factor=3,
                 repetition_time=1.0,
                 hierarchical=True,
                 motion_directory='motion_correction'):
        self.reference_volume = reference_volume
        self.registration_method = registration_method
        self.registration_type = registration_type
        self.mask = mask
        self.verbose = verbose
        self.interleave_factor = interleave_factor
        self.repetition_time = repetition_time
        self.hierarchical = hierarchical
        self._volumes = []
        self._transformations = {
            'v2v': {},
            'ss2v': {},  
            's2v': {}   
        }
        self._warped_volumes = []
        self._warped_slices = {}
        self._warped_slice_sets = {}
        self.motion_corrected_bold = motion_corrected_nii
        self.motion_directory = motion_directory
        self.external = ExternalToolRunner()
        
    def _get_filenames(self, filepath):
        path, filename = os.path.split(filepath)
        base, _ = os.path.splitext(filename)
        base, _ = os.path.splitext(base)
        return path, base
    
    def _create_reference(self, bold_nii, mask_nii, ref_nii=None):
        img, affine, header = load_nifti_data(bold_nii)
        mask = load_nifti_data(mask_nii)[0]
        
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
        ref_header = header.copy()
        ref_header['dim'][0] = 3 
        ref_header['dim'][4] = 1  
        reference = nib.Nifti1Image(ref_img, affine, ref_header)
        if ref_nii:
            nib.save(reference, ref_nii)
        else:
            path, base = self._get_filenames(bold_nii)
            ref_nii = os.path.join(path, f"{base}_reference.nii.gz")
            nib.save(reference, ref_nii)
        print("Reference image generated and saved.")
        self.reference_volume = ref_nii
        
    def _initialize_registration(self, registration_method):
        if registration_method == 'SimpleITK':
            reg = sitk.ImageRegistrationMethod()
            reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsGradientDescent(
                learningRate=1.0, 
                numberOfIterations=100, 
                convergenceMinimumValue=1e-6, 
                convergenceWindowSize=10
                )
            reg.SetOptimizerScalesFromPhysicalShift()
            return reg
        
        # Placeholder for other registration methods
        elif registration_method == 'RegAladin':
            pass
        elif registration_method == 'antsRegistration':
            pass
        elif registration_method == 'flirt':
            pass
        else:
            raise ValueError("Unsupported registration method.")
        
    def _save_transformations(self, transformations, suffix):
        if not os.path.exists(self.motion_directory):
            os.makedirs(self.motion_directory)
    
        for volume_name, transform in transformations.items():
            transform_filename = os.path.join(self.motion_directory, f"{volume_name}_{suffix}.tfm")
        
            if isinstance(transform, sitk.Transform):
                sitk.WriteTransform(transform, transform_filename)
            elif isinstance(transform, ants.ANTsTransform):
                pass
            else:
                raise ValueError("Unsupported transformation type for saving.")
        
            if self.verbose:
                print(f"Transformation for {volume_name} saved as {transform_filename}")
            
    def _split_slices_based_on_interleave(self, slices):
        sub_volumes = []
        num_slices = len(slices)
        for i in range(0, num_slices, self.interleave):
            sub_volumes.append(slices[i:i+self.interleave])
        return sub_volumes
    
    def _merge_corrected_volumes(self):
        if not self._warped_volumes:
            raise ValueError("There are no corrected volumes to merge.")
        corrected_vols = [sitk.ReadImage(vol) for vol in self._warped_volumes]
        _, base_name = self._get_filenames(self._warped_volumes[0])
        input_name = '_'.join(base_name.split('_')[:-2])
        
        corrected_4d = sitk.JoinSeries(corrected_vols)
        original_spacing = corrected_vols[0].GetSpacing()
        new_spacing = tuple(original_spacing) + (float(self.repetition_time),)
        corrected_4d.SetSpacing(new_spacing)
        
        if self.motion_corrected_bold:
            output_4d_path = self.motion_corrected_bold
        else:
            output_4d_path = os.path.join(self.motion_directory, input_name + '_mc.nii.gz')
        sitk.WriteImage(corrected_4d, output_4d_path)
        
        print(f"Motion corrected BOLD image saved at: {output_4d_path}")
        return output_4d_path
    
    def _run_registration(self, fixed_image, moving_image, mask, method, type, output_path):
        _, moving_name = self._get_filenames(moving_image)
        
        if method == 'SimpleITK':
            fixed_image_sitk = sitk.ReadImage(fixed_image)
            moving_image_sitk = sitk.ReadImage(moving_image)
            mask_image_sitk = sitk.ReadImage(mask) if mask else None
            if type == 'rigid':
                initial_transform = sitk.CenteredTransformInitializer(fixed_image_sitk, 
                                                                  moving_image_sitk, 
                                                                  sitk.Euler3DTransform(), 
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            elif type == 'affine':
                initial_transform = sitk.CenteredTransformInitializer(fixed_image_sitk, 
                                                                  moving_image_sitk, 
                                                                  sitk.AffineTransform(fixed_image_sitk.GetDimension()), 
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            else:
                raise ValueError("Unsupported registration type. Use 'rigid' or 'affine'.")    
            
            reg = self._initialize_registration(method)
            reg.SetInitialTransform(initial_transform, inPlace=False)
            if mask_image_sitk:
                reg.SetMetricFixedMask(mask_image_sitk)

            try:
                final_transform = reg.Execute(fixed_image_sitk, moving_image_sitk)
            except Exception as e:
                print(f"Registration failed: {e}")
                return None, None

            print("Optimizer stop condition: {0}".format(reg.GetOptimizerStopConditionDescription()))
            print("Final metric value: {0}".format(reg.GetMetricValue()))

            moving_resampled = sitk.Resample(
                moving_image_sitk, 
                fixed_image_sitk, 
                final_transform, 
                sitk.sitkLinear, 
                0.0, 
                moving_image_sitk.GetPixelID()
                )
            moving_resampled_nii = os.path.join(output_path, f"{moving_name}_warped.nii.gz")
            sitk.WriteImage(moving_resampled, moving_resampled_nii)
            return moving_resampled_nii, final_transform
    
    def _volume_to_volume_registration(self):    
        for i, volume in enumerate(self._volumes):
            txt = f"Volume-to-Volume Registration -- Volume {i + 1}/{len(self._volumes)}"
            if self.verbose:
                print(txt)
            if not os.path.isfile(volume):
                raise ValueError(f"Volume path is not a valid file: {volume}")
            _, volume_name = self._get_filenames(volume)
            warped_volume, transform_sitk = self._run_registration(self.reference_volume, volume, self.mask, self.registration_method, self.registration_type, self.motion_directory)
            self._warped_volumes.append(warped_volume)
            self._transformations['v2v'][volume_name] = transform_sitk           

        self._save_transformations(self._transformations['v2v'], 'v2v')

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
            self._merge_corrected_volumes()

        print("Registration completed.")