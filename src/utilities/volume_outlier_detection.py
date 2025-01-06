import os
import numpy as np
import nibabel as nib
from scipy.stats import norm

class OutlierDetection:
    def __init__(self, 
                 bold_nii, 
                 mask_nii=None, 
                 method='pyToutcount', 
                 normalize=False, 
                 dvars_thr =None, 
                 fraction_thr=0.03,
                 polort=0,
                 qthr = 0.001):
        self.bold_nii = bold_nii
        self.mask_nii = mask_nii
        self.method = method
        self.polort = polort
        self.qthr = qthr
        self.normalize = normalize
        self.dvars_thr = dvars_thr
        self.fraction_thr = fraction_thr
        self.img, self.mask = self._load_data()
        self.voxel_timeseries = self._prepare_timeseries()
        
    def _load_data(self):
        bold_img = nib.load(self.bold_nii)
        img = bold_img.get_fdata()
        
        if self.mask_nii:
            mask_img = nib.load(self.mask_nii)
            mask = mask_img.get_fdata().astype(bool)
        else:
            mask = np.ones(img.shape[:3], dtype=bool)
    
        return img, mask
    
    def _prepare_timeseries(self):
        return self.img[self.mask].reshape(-1, self.img.shape[-1])
    
    def _get_filenames(self, filepath):
        path, filename = os.path.split(filepath)
        base, _ = os.path.splitext(filename)
        base, _ = os.path.splitext(base)
        return path, base
    
    def detrend(self):
        if self.polort > 0:
            x = np.arange(self.img.shape[-1])
            for degree in range(self.polort + 1):
                trend = np.polyval(np.polyfit(x, self.voxel_timeseries.T, degree), x)
                self.voxel_timeseries -= trend.T
        else:
            median_trend = np.median(self.voxel_timeseries, axis=1, keepdims=True)
            self.voxel_timeseries -= median_trend
    
    def calculate_mad(self):
        return np.median(np.abs(self.voxel_timeseries - np.median(self.voxel_timeseries, axis=1, keepdims=True)), axis=1)
    
    def detect_outliers_afni(self):
        q = self.qthr
        mad = self.calculate_mad()
        
        N = self.img.shape[-1]
        alpha = norm.ppf(1 - q / N)
        threshold = alpha * np.sqrt(np.pi / 2) * mad
        
        outliers = np.abs(self.voxel_timeseries - np.median(self.voxel_timeseries, axis=1, keepdims=True)) > threshold[:, np.newaxis]
        
        return outliers
    
    def normalize_timeseries(self):
        mean_ts = np.mean(self.voxel_timeseries, axis=1, keepdims=True)
        std_ts = np.std(self.voxel_timeseries, axis=1, keepdims=True)
        self.voxel_timeseries = (self.voxel_timeseries - mean_ts) / std_ts
    
    def calculate_dvars(self):
        if self.normalize:
            self.normalize_timeseries()
            
        diff_ts = np.diff(self.voxel_timeseries, axis=1)
        dvars = np.sqrt(np.mean(diff_ts**2, axis=0))
        return dvars
    
    def mark_outlier_volumes_afni(self, outliers):
        fraction = self.fraction_thr
        outlier_fractions = np.mean(outliers, axis=0)
        volume_outliers = outlier_fractions > fraction
        outlier_indices = np.where(volume_outliers)[0]
        
        return outlier_indices, volume_outliers

    def mark_outlier_volumes_dvars(self, dvars, dvars_threshold):
        volume_outliers = dvars > dvars_threshold
        outlier_indices = np.where(volume_outliers)[0]
        return outlier_indices, volume_outliers
    
    def _save_outlier_indices(self, outlier_indices):
        path, base = self._get_filenames(self.bold_nii)
        filename = f"{base}_outvols_{self.method}.txt"
        output_file = os.path.join(path, filename)
        np.savetxt(output_file, outlier_indices, fmt='%d')
        
        print(f"Volume outlier indices saved to {output_file}")
    
    def run(self):
        if self.method == 'pyToutcount':
            self.detrend()
            outliers = self.detect_outliers_afni()
            outlier_indices, volume_outliers = self.mark_outlier_volumes_afni(outliers)
        elif self.method == 'dvars':
            dvars = self.calculate_dvars()
            if self.dvars_thr is None:
                dvars_threshold = np.mean(dvars) + 2 * np.std(dvars)  # Default threshold: mean + 2*std
            else:
                dvars_threshold = self.dvars_thr
            outlier_indices, volume_outliers = self.mark_outlier_volumes_dvars(dvars, dvars_threshold)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._save_outlier_indices(outlier_indices)
        
        return outlier_indices, volume_outliers
        
    