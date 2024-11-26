import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import subprocess
from loguru import logger
from src.utilities.external_running import ExternalToolRunner

class N4BiasFieldCorrection:
    def __init__(self, bspline_fitting_distance=300, shrink_factor=3, n_iterations=[50, 50, 30, 20], 
                 histogram_sharpening_fwhm=0.15, wiener_noise=0.01, num_histogram_bins=200):        
        """
        Initialize the N4BiasFieldCorrection class with parameters for N4 bias field correction.
        see the helper of N4N4BiasFieldCorrection function in ANTs for details
        """
        self.bspline_fitting_distance = bspline_fitting_distance
        self.shrink_factor = shrink_factor
        self.n_iterations = n_iterations
        self.histogram_sharpening_fwhm = histogram_sharpening_fwhm
        self.wiener_noise = wiener_noise
        self.num_histogram_bins = num_histogram_bins
        self.external = ExternalToolRunner()
        
    #def run_command(self, command):
    #    try:
    #        subprocess.run(command, shell=True, env=env, check=True)
    #        print(f"Command executed successfully: {command}")
    #    except subprocess.CalledProcessError as e:
    #        print(f"Error: {e}")
    #        raise
        
    def _make_4d_mask(self, mask_3d_nii, num_vols):
        path, _ = os.path.split(mask_3d_nii)
        mask_img = nib.load(mask_3d_nii)
        img = mask_img.get_fdata()
        img_4d = np.repeat(img[..., np.newaxis], num_vols, axis=3)
        img_4d_nii = nib.Nifti1Image(img_4d, mask_img.affine, mask_img.header)
        tmp_mask_4d_path = os.path.join(path, "tmp_mask_4d.nii.gz")
        nib.save(img_4d_nii, tmp_mask_4d_path)
        return tmp_mask_4d_path
            
    def run_correction(self, input_nii, mask_nii, output_nii, output_bias_nii=None, method='ANTs4D'):
        """
        Perform N4 bias field correction on the input image and save the corrected image and bias field.

        Parameters:
        - input_nii (str): Path to the input NIfTI image file.
        - mask_nii (str): Path to the mask NIfTI image file.
        - output_nii (str): Path to save the corrected image file.
        - output_bias_nii (str): Path to save the bias field image file.
        - method (str): Method to use for correction ('ANTs4D' or 'sitk3D').
        """
        image = sitk.ReadImage(input_nii, sitk.sitkFloat32)
        num_vols = image.GetSize()[3]
        mask_img = nib.load(mask_nii)
        path, _ = os.path.split(mask_nii)
        
        if method == "ANTs4D":
            
            if len(mask_img.shape) == 3:
                mask_nii_4d = self._make_4d_mask(mask_nii, num_vols)
            elif len(mask_img.shape) == 4:
                mask_nii_4d = mask_nii
            else:
                raise ValueError("Mask is neither 3D nor 4D.")
            
            ants_command = (
                f"N4BiasFieldCorrection -d 4 "
                f"-i {input_nii} "
                f"-x {mask_nii_4d} "
                f"-s {self.shrink_factor} "
                f"-c {'x'.join(map(str, self.n_iterations))} "
                f"-b [ {self.bspline_fitting_distance}, 3 ] "
                f"-t [ {self.histogram_sharpening_fwhm}, {self.wiener_noise}, {self.num_histogram_bins} ] "
                f"-o [ {output_nii}, {output_bias_nii} ]"
            )  
                  
            self.external.run_command(ants_command, tool='ants')
            
            if os.path.exists(os.path.join(path, "tmp_mask_4d.nii.gz")):
                os.remove(os.path.join(path, "tmp_mask_4d.nii.gz"))
            
        elif method == "sitk3D":
            bfc_corrector = sitk.N4BiasFieldCorrectionImageFilter()
            bfc_vols = []
            for vol in range(num_vols):
                print(f"Processing volume {vol + 1}/{num_vols} with SimpleITK filter...")
                volume = sitk.Extract(image, size=(image.GetSize()[0], image.GetSize()[1], image.GetSize()[2], 0), index=(0, 0, 0, vol))
                mask = sitk.ReadImage(mask_nii, sitk.sitkUInt8)
                corrected_volume = bfc_corrector.Execute(volume, mask)
                bfc_vols.append(corrected_volume)
                
            corrected_image = sitk.JoinSeries(bfc_vols)
            sitk.WriteImage(corrected_image, output_nii)
            
            if output_bias_nii:
                bias_fields = [bfc_corrector.GetLogBiasFieldAsImage(volume) for volume in bfc_vols]
                bias_field_image = sitk.JoinSeries(bias_fields)
                sitk.WriteImage(bias_field_image, output_bias_nii)
        else:
            raise ValueError("Invalid method selected. Choose either 'ANTs4D' or 'sitk3D'.")