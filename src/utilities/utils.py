# Athena Taymourtash, 2024
# athena.taymourtash@meduniwien.ac.at

import os, sys, glob
import shutil
import subprocess
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk


import seaborn as sns
import matplotlib.pyplot as plt


def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        raise
    
def load_nifti_data(input_nii):
    input_img = nib.load(input_nii)
    img = input_img.get_fdata()
    affine = input_img.affine
    header = input_img.header
    return img, affine, header

def dilate_mask(mask_nii, mask_dil_nii=None, kernel_radius=None, kernel_type=sitk.sitkBall):
    mask = sitk.ReadImage(mask_nii)
    if kernel_radius is not None:
        kernel_radius = [int(kernel_radius)] * mask.GetDimension()
    mask_dilated = sitk.BinaryDilate(mask, kernelRadius=kernel_radius, kernelType=kernel_type)
    if mask_dil_nii:
        sitk.WriteImage(mask_dilated, mask_dil_nii)
        return mask_dil_nii
    return mask_dilated

def crop_image(reference_nii, mask_nii, cropped_nii):
    cal_roi = f"fslstats {mask_nii} -w"
    bounding_box = run_command(cal_roi)
    crp_roi = f"fslroi {reference_nii} {cropped_nii} {bounding_box}"
    run_command(crp_roi)
    return cropped_nii

def get_num_vols(input_img):
    return input_img.shape[3]

def get_repetition_time(bold_nii):
    bold_img = nib.load(bold_nii)
    return bold_img.header['pixdim'][4]

def make_4d_mask(mask_nii, num_vols):
    mask_img = nib.load(mask_nii)
    img = mask_img.get_fdata()
    img_4d = np.repeat(img[..., np.newaxis], num_vols, axis=3)
    img_4d_nii = nib.Nifti1Image(img_4d, mask_img.affine, mask_img.header)
    return img_4d_nii

def split_to_3d(input_nii, prefix):
    return input_nii
    
def apply_affine_transformation(reference_img_path, seg_img_path, affine_transform_path, output_img_path):
    reference_img = sitk.ReadImage(reference_img_path)
    seg_img = sitk.ReadImage(seg_img_path)

    affine_transform = sitk.ReadTransform(affine_transform_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetTransform(affine_transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_img = resampler.Execute(seg_img)

    sitk.WriteImage(resampled_img, output_img_path)    
       
def del_dummy_scans(bold_img, nd=4):
    censored_img = np.delete(bold_img,np.arange(0,nd),3)
    return censored_img

def get_derivative(timeseries_1d):
    deriv = np.zeros(timeseries_1d.shape)
    deriv[1:] = timeseries_1d[1:] - timeseries_1d[:-1]
    return deriv

def save_to_h5py(h5f, dataset_name, timeseries, derivative):
    try:
        h5f.create_dataset(dataset_name, data=timeseries)
        h5f.create_dataset(dataset_name + '_deriv', data=derivative)
    except:
        if dataset_name in h5f:
            del h5f[dataset_name]
        if (dataset_name + '_deriv') in h5f:
            del h5f[dataset_name + '_deriv']
        h5f.create_dataset(dataset_name, data=timeseries)
        h5f.create_dataset(dataset_name + '_deriv', data=derivative)
        
def get_subject_id(file_path):
    subject_id = os.path.basename(os.path.dirname(file_path))
    return subject_id

def get_filenames(filepath):
        path, filename = os.path.split(filepath)
        base, _ = os.path.splitext(filename)
        base, _ = os.path.splitext(base)
        return path, base
    


    



