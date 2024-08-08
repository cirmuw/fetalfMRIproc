import os
import numpy as np
import nibabel as nib
from src.utilities.utils import load_nifti_data, get_filenames

def generate_reference(bold_nii, mask_nii, ref_nii=None):
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
            path, base = get_filenames(bold_nii)
            ref_nii = os.path.join(path, f"{base}_reference.nii.gz")
            nib.save(reference, ref_nii)
        print("Reference image generated and saved.")
        return ref_nii    
    
