import os
import numpy as np
import nibabel as nib
from scipy.signal import detrend
from sklearn.decomposition import PCA

class NuisanceRegressor:
    def __init__(self, bold_nii, mask_nii=None, seg_nii=None, motion_dir=None,
                 wm_labels=None, csf_labels=None):
        self.bold_nii = bold_nii
        self.mask_nii = mask_nii
        self.seg_nii = seg_nii
        self.motion_dir = motion_dir
        self.wm_labels = wm_labels if wm_labels is not None else []
        self.csf_labels = csf_labels if csf_labels is not None else []
        self.regressors = {}
        self.img, self.affine, self.header = self._load_bold_data()
        self.seg = self._load_segmentation_data()
        self.mask = self._load_mask_data()
        
        
    def _load_bold_data(self):
        bold_img = nib.load(self.bold_nii)
        img = bold_img.get_fdata()
        affine = bold_img.affine
        header = bold_img.header
        return img, affine, header
    
    def _load_mask_data(self):
        if self.seg_nii:
            mask = self.seg > 0
        elif self.mask_nii:
            mask_img = nib.load(self.mask_nii)
            mask = mask_img.get_fdata().astype(bool)
        else:
            mask = np.ones(self.img.shape[:3], dtype=bool)
        return mask
    
    def _load_segmentation_data(self): 
        if self.seg_nii:
            seg_img = nib.load(self.seg_nii)
            seg = seg_img.get_fdata().astype(int)
        else:
            seg = None
        return seg
    
    def extract_global_signal(self):
        masked_img = self.img[self.mask]
        global_signal = masked_img.mean(axis=0)
        self.regressors['global_signal'] = global_signal
        return global_signal
    
    def extract_wm_signal(self):
        if self.seg is None and self.wm_labels: 
            wm_mask = np.isin(self.seg, self.wm_labels)
            wm_mask = wm_mask[self.mask]
            wm_masked_img = self.img[wm_mask]
            wm_signal = wm_masked_img.mean(axis=0)
            self.regressors['wm_signal'] = wm_signal
            return wm_signal
        else:
            raise ValueError("Segmentation is not available or no WM labels provided.")
        

    def extract_csf_signal(self):
        if self.seg is None and self.csf_labels:
            csf_mask = np.isin(self.seg, self.csf_labels)
            csf_mask = csf_mask[self.mask]
            csf_masked_img = self.img[csf_mask]
            csf_signal = csf_masked_img.mean(axis=0)
            self.regressors['csf_signal'] = csf_signal
            return csf_signal
        else:
            raise ValueError("Segmentation is not available or no CSF labels provided.")
        
    
    
        