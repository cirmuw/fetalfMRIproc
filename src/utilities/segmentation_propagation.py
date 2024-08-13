import os
import numpy as np
import SimpleITK as sitk


class SegmentationPropagation:
    def __init__(self, transformation, template, is_inverse=None, output_directory=None):
        self.transformation = transformation
        self.template = template
        self.is_inverse = is_inverse
        self.output_directory = output_directory
        
    def _get_ext(transformation):
        return os.path.splitext(transformation)[1]
    
    def _get_inverse(self):
        _, base = os.path.split(self.transformation)
        basename, ext = os.path.splitext(base)
        inverse_transform_path = os.path.join(self.output_directory, basename + '_inversed' + ext)
        
        if ext == '.txt':
            transform = sitk.ReadTransform(self.transformation)
            if transform.IsLinear():
                inverse_transform = transform.GetInverse()
                sitk.WriteTransform(inverse_transform, inverse_transform_path)
                print(f"Inverse transformation saved to {inverse_transform_path}")
       
        elif ext == '.mat':
            cmd = 'convert_xfm -omat %s -inverse %s > /dev/null' % (inverse_transform_path, self.transformation)
            os.system(cmd)
            
        return inverse_transform_path
        
    
    def _propagate_labels(self, seg_nii):
        ext = self._get_ext(self.transformation)
        reference_space = self.template
        seg_space = seg_nii
        _, base = os.path.split(seg_nii)
        seg_resampled_path = os.path.join(self.output_directory, base[:-7] + '_resampled.nii.gz') 
        
        if self.is_inverse:
            transform_file = self._get_inverse()
        else:
            transform_file = self.transformation    
        
        if ext == '.txt':
            reference_sitk = sitk.ReadImage(reference_space)
            seg_sitk = sitk.ReadImage(seg_nii)
            transform = sitk.ReadTransform(transform_file)
            resampled_sitk = sitk.Resample(
                seg_sitk,
                reference_sitk,
                transform,
                sitk.sitkNearestNeighbor,  # Nearest neighbor interpolation to preserve labels
                0,  
                seg_sitk.GetPixelID()  
                )
            sitk.WriteImage(resampled_sitk, seg_resampled_path)
            print(f"Resampled segmentation saved to {self.output_directory}")
        
        elif ext == '.mat':
            cmd = 'flirt -interp nearestneighbour -in %s -ref %s -applyxfm -init %s \
                -out %s > /dev/null' % (seg_nii, reference_space, transform_file, seg_resampled_path)
            os.system(cmd)
            print(f"Resampled segmentation saved to {seg_resampled_path}")
            
        return seg_resampled_path
        
    