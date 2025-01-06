import os
import subprocess
import numpy as np
from scipy.stats import median_abs_deviation
import nibabel as nib

class SpikeRejection:
    def __init__(self, bold_nii, mask_nii=None, method = 'Lfit', c1=2.5, c2=4.0, corder=None):
        self.bold_nii = bold_nii
        self.mask_nii = mask_nii
        self.c1 = c1
        self.c2 = c2
        self.corder = corder
        self.method = method
        if self.method != 'afni':
            self.img, self.mask, self.affine, self.header = self._load_data()
            self.voxel_timeseries = self._prepare_timeseries()
        
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
    
    def _prepare_timeseries(self):
        return self.img[self.mask].reshape(-1, self.img.shape[-1])
    
    def _get_filenames(self, filepath):
        path, filename = os.path.split(filepath)
        base, _ = os.path.splitext(filename)
        base, _ = os.path.splitext(base)
        return path, base
    
    def _run_command(self, command):
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            raise
        
    def despike_afni(self, despiked_nii_output):
        despiking_command = f"3dDespike -NEW -nomask -prefix {despiked_nii_output} {self.bold_nii}"
        self.run_command(despiking_command)
    
    def _fit_curve(self, y):
        x = np.arange(y.size)
        if self.corder is None:
            L = y.size // 30
        else:
            L = self.corder
        
        coefs = np.polyfit(x, y, 2)
        curve = np.polyval(coefs, x)
        
        for k in range(1, L+1):
            curve += np.sin(2 * np.pi * k * x / y.size) + np.cos(2 * np.pi * k * x / y.size)
        
        return curve

    def _calculate_residuals(self, y, curve):
        return y - curve

    def _calculate_sigma(self, residuals):
        return median_abs_deviation(residuals)

    def _replace_spikes(self, y, curve, sigma):
        s = (y - curve) / sigma
        spikes = s > self.c1
        
        y_spike_corrected = np.copy(y)
        y_spike_corrected[spikes] = curve[spikes] + sigma * (self.c1 + (self.c2 - self.c1) * np.tanh((s[spikes] - self.c1) / (self.c2 - self.c1)))
        
        return y_spike_corrected, spikes

    def despike_voxel_timeseries(self, y):
        curve = self._fit_curve(y)
        residuals = self._calculate_residuals(y, curve)
        sigma = self._calculate_sigma(residuals)
        despike_ts, spikes = self._replace_spikes(y, curve, sigma)
        return despike_ts, spikes

    def despike(self, despiked_nii_output=None, spike_mask_nii_output=None):
        
        if despiked_nii_output is None or spike_mask_nii_output is None:
            path, base = self._get_filenames(self.bold_nii)
            
            if despiked_nii_output is None:
                despiked_nii_output = os.path.join(path, f"{base}_dspk.nii.gz")
            
            if spike_mask_nii_output is None:
                spike_mask_nii_output = os.path.join(path, f"{base}_dspk_mask4d.nii.gz")
        
        if self.method == 'afni':
            self.despike_afni(despiked_nii_output)
            # AFNI does not produce a spike mask, so we won't generate one here.
            return despiked_nii_output, None
        
        elif self.method == 'Lfit':
            despiked_img = np.copy(self.img)
            spikes_mask = np.zeros_like(self.img, dtype=bool)
        
            for voxel_index in range(self.voxel_timeseries.shape[0]):
                y = self.voxel_timeseries[voxel_index]
                despiked_ts, spikes = self.despike_voxel_timeseries(y)
                despiked_img[self.mask][voxel_index] = despiked_ts
                spikes_mask[self.mask][voxel_index] = spikes

            despiked_img_nib = nib.Nifti1Image(despiked_img, self.affine, self.header)
            nib.save(despiked_img_nib, despiked_nii_output)

            inverted_spikes_mask = np.logical_not(spikes_mask).astype(np.uint8)

            spikes_img_nib = nib.Nifti1Image(inverted_spikes_mask, self.affine, self.header)
            nib.save(spikes_img_nib, spike_mask_nii_output)
            return spikes_img_nib, spike_mask_nii_output
        
        else:
            raise ValueError("Unsupported method of despiking. Use 'Lfit' or 'afni'.")
            
            
        
        