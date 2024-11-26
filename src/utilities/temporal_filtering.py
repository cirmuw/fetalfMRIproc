import os
import math
import numpy as np
import nibabel as nib
from scipy.signal import butter, filtfilt

class TemporalFiltering:
    def __init__(self, bold_nii, mask_nii, repetition_time, lowcut=None, highcut=None, filter_type='dct', order=5):
        """
        Initialize the TemporalFiltering class.

        bold_nii: 4D fMRI image
        mask_nii: 3D mask
        TR: Repetition time (in seconds)
        lowcut: Lower frequency cutoff for bandpass filter
        highcut: Upper frequency cutoff for bandpass filter
        filter_type: Type of filter to use ('butterworth' or 'dct')
        order: Order of the Butterworth filter (only used for 'butterworth' type)
        """
        self.bold_nii = bold_nii
        self.mask_nii = mask_nii
        self.img, self.mask, self.affine, self.header = self._load_data()
        self.repetition_time = repetition_time
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_type = filter_type
        self.order = order
        
    def _load_data(self):
        bold_img = nib.load(self.bold_nii)
        img = bold_img.get_fdata()
        affine = bold_img.affine
        header = bold_img.header
        
        if self.mask_nii:
            mask_img = nib.load(self.mask_nii)
            mask = mask_img.get_fdata().astype(bool)
        else:
            mask = np.ones(img.shape[:3], dtype=bool)
        
        return img, mask, affine, header
    
    def _get_filenames(self, filepath):
        path, filename = os.path.split(filepath)
        base, _ = os.path.splitext(filename)
        base, _ = os.path.splitext(base)
        return path, base

    def _butter_bandpass(self):
        nyquist = 0.5 / self.repetition_time
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def _butter_bandpass_filter(self):
        b, a = self._butter_bandpass()
        filtered_data = filtfilt(b, a, self.img, axis=-1)
        filtered_data[~self.mask] = self.img[~self.mask]
        return filtered_data

    def _discrete_cosine(self):
        nvol = self.img.shape[-1]
        n = np.arange(0, nvol)

        if self.lowcut is None:
            range_lowcut = []
        else:
            range_lowcut = np.arange(math.ceil(2 * (nvol * self.repetition_time) / (1 / self.lowcut)), nvol)

        if self.highcut is None:
            range_highcut = []
        else:
            range_highcut = np.arange(1, math.ceil(2 * (nvol * self.repetition_time) / (1 / self.highcut)))

        K = np.hstack((range_highcut, range_lowcut))

        if len(K) > 0:
            C = np.sqrt(2 / nvol) * np.cos(np.pi * (2 * n[:, None] + 1) * K / (2 * nvol))
        else:
            C = np.empty((nvol, 0))

        return C

    def _apply_discrete_cosine_filter(self):
        C = self._discrete_cosine()
        data_detrended = np.copy(self.img)
        mean_data = np.mean(self.img, axis=-1, keepdims=True)
        data_centered = self.img - mean_data
        data_detrended = data_centered - np.dot(np.dot(data_centered, C), C.T)
        data_detrended += mean_data
        data_detrended[~self.mask] = self.img[~self.mask]
        return data_detrended

    def apply_filter(self, filtered_nii_output=None):
        path, base = self._get_filenames(self.bold_nii)
        if filtered_nii_output is None:
                filtered_nii_output = os.path.join(path, f"{base}_tf.nii.gz")
        
        if self.filter_type == 'butterworth':
            detrended_img = self._butter_bandpass_filter()
        elif self.filter_type == 'dct':
            detrended_img = self._apply_discrete_cosine_filter()
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        detrended_img_nib = nib.Nifti1Image(detrended_img, self.affine, self.header)
        nib.save(detrended_img_nib, filtered_nii_output)
        return filtered_nii_output
        
    