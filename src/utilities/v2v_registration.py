import os
import nibabel as nib
import numpy as np
from time import time
from src.utilities.utils import run_command
from src.utilities.definitions import NIFTYREG_PATH


def fmri_motion_correction(input_nii, reference_nii, mask_nii, output_nii, save_folder_path=None):
    """
    Perform gross motion correction on a 4D fMRI image.
    
    Parameters:
    input_nii (str): Path to the input 4D fMRI image.
    reference_nii (str): Path to the reference image.
    mask_nii (str): Path to the mask image.
    output_nii (str): Path to the output motion-corrected 4D image.
    save_folder_path (str): Path to save intermediate files. Defaults to a temporary folder.
    
    Returns:
    str: Path to the motion-corrected 4D fMRI image.
    """
    time_0 = time()

    # Create the folder where to save the registration output
    tmp_folder = './tmp'
    if save_folder_path is None: 
        save_folder_path = tmp_folder

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    volumes = split_to_3d(input_nii, save_folder_path)
    
    mask_4d = prepare_mask(mask_nii, input_nii, save_folder_path)

    registered_volumes = []
    for vol in volumes:
        vol_name = os.path.basename(vol).split('.nii')[0]
        cropped_vol = os.path.join(save_folder_path, f"{vol_name}_CROPPED.nii.gz")
        crop_image(vol, mask_4d, cropped_vol)
        
        registered_vol = os.path.join(save_folder_path, f"{vol_name}_mc.nii.gz")
        register_volume(reference_nii, cropped_vol, mask_4d, registered_vol)
        registered_volumes.append(registered_vol)
    
    merge_to_4d(registered_volumes, output_nii)
    
    if save_folder_path == tmp_folder and os.path.exists(tmp_folder):
        os.system(f'rm -r {tmp_folder}')

    duration = int(time() - time_0) 
    minutes = duration // 60
    seconds = duration - minutes * 60

    return output_nii


def split_to_3d(input_nii, save_folder):
    """
    Split a 4D NIfTI image into separate 3D volumes.
    
    Parameters:
    input_nii (str): Path to the input 4D NIfTI image.
    save_folder (str): Path to save the split 3D volumes.
    
    Returns:
    list: List of paths to the split 3D volumes.
    """
    img = nib.load(input_nii)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    volume_paths = []
    for i in range(data.shape[3]):
        vol_data = data[..., i]
        vol_img = nib.Nifti1Image(vol_data, affine, header)
        vol_path = os.path.join(save_folder, f"vol_{i:04d}.nii.gz")
        nib.save(vol_img, vol_path)
        volume_paths.append(vol_path)
    
    return volume_paths


def prepare_mask(mask_nii, reference_nii, save_folder):
    """
    Prepare a 4D mask by repeating a 3D mask for each volume.
    
    Parameters:
    mask_nii (str): Path to the input 3D mask NIfTI image.
    reference_nii (str): Path to the reference 4D NIfTI image.
    save_folder (str): Path to save the 4D mask image.
    
    Returns:
    str: Path to the 4D mask image.
    """
    mask_img = nib.load(mask_nii)
    mask_data = mask_img.get_fdata()
    reference_img = nib.load(reference_nii)
    reference_shape = reference_img.shape
    mask_4d_data = np.repeat(mask_data[:, :, :, np.newaxis], reference_shape[3], axis=3)
    mask_4d_img = nib.Nifti1Image(mask_4d_data, mask_img.affine, mask_img.header)
    mask_4d_path = os.path.join(save_folder, "mask_4d.nii.gz")
    nib.save(mask_4d_img, mask_4d_path)
    
    return mask_4d_path


def crop_image(input_nii, mask_nii, output_nii):
    """
    Crop the input image using the bounding box of the mask.
    
    Parameters:
    input_nii (str): Path to the input NIfTI image.
    mask_nii (str): Path to the mask NIfTI image.
    output_nii (str): Path to the output cropped NIfTI image.
    """
    fslstats_cmd = f"fslstats {mask_nii} -w"
    bounding_box = run_command(fslstats_cmd)
    fslroi_cmd = f"fslroi {input_nii} {output_nii} {bounding_box}"
    run_command(fslroi_cmd)


def register_volume(reference_nii, input_nii, mask_nii, output_nii):
    """
    Register a volume to a reference image using NiftyReg.
    
    Parameters:
    reference_nii (str): Path to the reference NIfTI image.
    input_nii (str): Path to the input NIfTI volume.
    mask_nii (str): Path to the mask NIfTI image.
    output_nii (str): Path to the output registered NIfTI image.
    """
    tmp_path = os.path.join(os.path.dirname(output_nii), "transformation.txt")
    reg_cmd = f"{NIFTYREG_PATH}/reg_aladin -ref {reference_nii} -flo {input_nii} -rmask {mask_nii} -affDirect -res {output_nii} -aff {tmp_path}"
    run_command(reg_cmd)
    resample_cmd = f"{NIFTYREG_PATH}/reg_resample -ref {reference_nii} -flo {input_nii} -trans {tmp_path} -inter 1 -res {output_nii}"
    run_command(resample_cmd)


def merge_to_4d(volume_paths, output_nii):
    """
    Merge separate 3D volumes into a single 4D NIfTI image.
    
    Parameters:
    volume_paths (list): List of paths to the 3D NIfTI volumes.
    output_nii (str): Path to the output 4D NIfTI image.
    """
    merged_cmd = f"fslmerge -t {output_nii} " + " ".join(volume_paths)
    run_command(merged_cmd)