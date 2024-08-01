import os
import nibabel as nib
import SimpleITK as sitk
from utils import get_num_vols, make_4d


def n4_bias_field_correction(input_nii, mask_nii, output_nii, output_bias_nii, 
                             convergence_threshold=1e-6, spline_order=3, 
                             wiener_filter_noise=0.11, bias_field_fwhm=0.15):
    
    image = sitk.ReadImage(input_nii, sitk.sitkFloat32)
    num_vols = image.GetSize()[3]
    mask_img = nib.load(mask_nii)
    
    if len(mask_img.shape) == 3:
        mask_nii_4d = make_4d(mask_img, num_vols)
    elif len(mask_img.shape) == 4:
        mask_nii_4d = mask_nii
    else:
        raise ValueError("Mask image is neither 3D nor 4D.")
    
    mask = sitk.ReadImage(mask_nii_4d, sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetConvergenceThreshold(convergence_threshold)
    corrector.SetSplineOrder(spline_order)
    corrector.SetWienerFilterNoise(wiener_filter_noise)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(bias_field_fwhm)
    
    corrected_image = corrector.Execute(image, mask)
    sitk.WriteImage(corrected_image, output_nii)
    
    bias_field = corrector.GetLogBiasFieldAsImage(image)
    sitk.WriteImage(bias_field, output_bias_nii)