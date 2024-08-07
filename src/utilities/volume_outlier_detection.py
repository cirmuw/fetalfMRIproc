import numpy as np
import nibabel as nib
from scipy.stats import norm

class OutlierDetection:
    def __init__(self, bold_nii, mask_nii=None, polort=0, normalize=False):
        self.bold_nii = bold_nii
        self.mask_nii = mask_nii
        self.polort = polort
        self.normalize = normalize
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
    
    def detrend(self):
        if self.polort > 0:
            x = np.arange(self.data.shape[-1])
            for degree in range(self.polort + 1):
                trend = np.polyval(np.polyfit(x, self.voxel_timeseries.T, degree), x)
                self.voxel_timeseries -= trend.T
        else:
            median_trend = np.median(self.voxel_timeseries, axis=1, keepdims=True)
            self.voxel_timeseries -= median_trend
    
    def calculate_mad(self):
        return np.median(np.abs(self.voxel_timeseries - np.median(self.voxel_timeseries, axis=1, keepdims=True)), axis=1)
    
    def detect_outliers_afni(self, q=0.001):
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
    
    def mark_outlier_volumes_afni(self, outliers, fraction=0.03):
        outlier_fractions = np.mean(outliers, axis=0)
        volume_outliers = outlier_fractions > fraction
        outlier_indices = np.where(volume_outliers)[0]
        
        return outlier_indices, volume_outliers

    def mark_outlier_volumes_dvars(self, dvars, dvars_threshold):
        volume_outliers = dvars > dvars_threshold
        outlier_indices = np.where(volume_outliers)[0]
        return outlier_indices, volume_outliers
    
    def run(self, method='3dToutcount', q=0.001, threshold_fraction=0.03, dvars_threshold=None):
        self.detrend()
        
        if method == '3dToutcount':
            outliers = self.detect_outliers_afni(q)
            outlier_indices, volume_outliers = self.mark_outlier_volumes_afni(outliers, threshold_fraction)
        elif method == 'dvars':
            dvars = self.calculate_dvars()
            if dvars_threshold is None:
                dvars_threshold = np.mean(dvars) + 2 * np.std(dvars)  # Default threshold: mean + 2*std
            outlier_indices, volume_outliers = self.mark_outlier_volumes_dvars(dvars, dvars_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outlier_indices, volume_outliers
        
    