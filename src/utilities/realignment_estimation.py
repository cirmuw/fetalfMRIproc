import os
import math
import numpy as np
import scipy.io
import nibabel as nib
from copy import copy

def get_motion_niftyreg(sub_id, n_vols, directory):
    """
    Compute motion parameters from NiftyReg outputs.
    
    Parameters:
    - sub_id: Subject identifier
    - n_vols: Number of volumes
    - directory: Directory containing motion parameter files
    
    Returns:
    - HMP: Array of motion parameters
    """
    HMP = np.zeros((n_vols, 6))
    
    for i in range(n_vols):
        file_path = os.path.join(directory, 'MOCO', 'Aladin', f'{sub_id}_bfc_vol{str(i).zfill(4)}.txt')
        motion_matrix = np.genfromtxt(file_path)
        
        HMP[i, :3] = motion_matrix[:3, 3]
        rotation_matrix = np.zeros((4, 4))
        rotation_matrix[:3, 0] = motion_matrix[:3, 0] / np.linalg.norm(motion_matrix[:, 0])
        rotation_matrix[:3, 1] = motion_matrix[:3, 1] / np.linalg.norm(motion_matrix[:, 1])
        rotation_matrix[:3, 2] = motion_matrix[:3, 2] / np.linalg.norm(motion_matrix[:, 2])
        rotation_matrix[3, 3] = 1
        
        HMP[i, 5] = math.degrees(math.atan(rotation_matrix[1, 0] / rotation_matrix[0, 0]))
        HMP[i, 4] = math.degrees(math.atan(-rotation_matrix[2, 0] / np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)))
        HMP[i, 3] = math.degrees(math.atan(rotation_matrix[2, 1] / rotation_matrix[2, 2]))
    
    return HMP

def get_motion_ants(sub_id, n_vols, directory):
    """
    Compute motion parameters from ANTs outputs.
    
    Parameters:
    - sub_id: Subject identifier
    - n_vols: Number of volumes
    - directory: Directory containing motion parameter files
    
    Returns:
    - HMP: Array of motion parameters
    """
    HMP = np.zeros((n_vols, 6))
    
    for i in range(n_vols):
        affine_mat_file1 = os.path.join(directory, 'MOCO', f'{sub_id}_bfc_vol{str(i).zfill(4)}0GenericAffine.mat')
        affine_mat1 = scipy.io.loadmat(affine_mat_file1)['AffineTransform_float_3_3']
        affine_mat1 = np.vstack((np.reshape(affine_mat1, (4, 3)).T, [0, 0, 0, 1]))
        
        affine_mat_file2 = os.path.join(directory, 'MOCO', f'{sub_id}_bfc_vol{str(i).zfill(4)}_2nd0GenericAffine.mat')
        affine_mat2 = scipy.io.loadmat(affine_mat_file2)['AffineTransform_float_3_3']
        affine_mat2 = np.vstack((np.reshape(affine_mat2, (4, 3)).T, [0, 0, 0, 1]))
        
        combined_affine_mat = np.dot(affine_mat2, affine_mat1)
        HMP[i, :3] = combined_affine_mat[:3, 3]
        
        rotation_matrix = np.zeros((4, 4))
        rotation_matrix[:3, 0] = combined_affine_mat[:3, 0] / np.linalg.norm(combined_affine_mat[:, 0])
        rotation_matrix[:3, 1] = combined_affine_mat[:3, 1] / np.linalg.norm(combined_affine_mat[:, 1])
        rotation_matrix[:3, 2] = combined_affine_mat[:3, 2] / np.linalg.norm(combined_affine_mat[:, 2])
        rotation_matrix[3, 3] = 1
        
        HMP[i, 5] = math.degrees(math.atan(rotation_matrix[1, 0] / rotation_matrix[0, 0]))
        HMP[i, 4] = math.degrees(math.atan(-rotation_matrix[2, 0] / np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)))
        HMP[i, 3] = math.degrees(math.atan(rotation_matrix[2, 1] / rotation_matrix[2, 2]))
    
    return HMP

def get_fd_rms(motion_params, head_motion):
    """
    Compute framewise displacement RMS.
    
    Parameters:
    - motion_params: Array of motion parameters
    - head_motion: Head motion in degrees
    
    Returns:
    - fd_rms: Framewise displacement RMS
    """
    motion_params_copy = copy(motion_params)
    motion_params_copy[:, 3:6] *= head_motion * (math.pi / 180)
    
    displacement = np.diff(motion_params_copy, axis=0)
    displacement = np.vstack(([0, 0, 0, 0, 0, 0], displacement))
    
    # Compute framewise displacement RMS
    fd_rms = np.sum(np.abs(displacement), axis=1)
    
    return fd_rms

def get_threshold(framewise_displacement):
    """
    Compute threshold for framewise displacement.
    
    Parameters:
    - framewise_displacement: Array of framewise displacement values
    
    Returns:
    - threshold: Framewise displacement threshold
    """
    q75, q25 = np.percentile(framewise_displacement, [75, 25])
    iqr = q75 - q25
    threshold = q75 + 1.5 * iqr
    
    return threshold
