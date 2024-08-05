import os 
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from src.utilities.utils import load_nifti_data

class MotionCorrection:
    def __init__(self, reference_volume=None):
        self.reference_volume = reference_volume
    
    def create_reference(self, bold_nii, mask_nii):
        
        bold_img = load_nifti_data(bold_nii)
        mask = load_nifti_data(mask_nii)
        
        if mask.shape != bold_img.shape[:3]:
            raise ValueError("Mask dimensions do not match the input image dimensions.")
        
        
        