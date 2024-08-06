##
# \file run_preprocessing_workflow.py
# \brief      Script to preprocess a raw in-utero fmri image (4d) and derive 
#             clean timeseries
#
# \author     Athena Taymourtash (athena.taymourtash@meduniwien.ac.at)
# \date       February 2021
#
import os
import sys
sys.path.append('/Users/athena/Documents/CIRHome/FetalRestingStatefMRI')
import numpy as np
from time import time
from loguru import logger
from argparse import ArgumentParser
from src.utilities.utils import *
from src.utilities import *
from src.utilities.bias_field_correction import N4BiasFieldCorrection
from src.utilities.base_motion_correction import MotionCorrection

def main():
    
    
    np.set_printoptions(precision=3)
    
    parser = ArgumentParser(description="Fetal rs-fMRI workflow", epilog='\n')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--input-mask', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--T2-to-bold', required=True, type=str)
    parser.add_argument('--v2v-method', required=False, type=str, default="RegAladin")
    parser.add_argument('--dummy-frames', required=False, type=int, default=0)
    parser.add_argument('--dilation-radius', required=False, type=int, default=5)
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    mask_path = args.input_mask
    logger.info('Input 4D fMRI: %s' % input_path)
    logger.info('Input 3D brain mask: %s' % mask_path)
    logger.info('Output folder: %s' % output_path)
    sid = get_subject_id(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    # ---------------------- Read Data (!TO-DO: modular) -------------
    
    dilation_radius=args.dilation_radius
    img = load_nifti_data(input_path)
    
    if args.dummy_frames:
        img = del_dummy_scans(img)
    
    num_vol = get_num_vols(img)
    mask_dilated = dilate_mask(mask_path, os.path.join(output_path, sid + '_mask_dilated.nii.gz'), dilation_radius)
    
    # ---------------------- Bias Field Correction -------------------
    bold_bfc = os.path.join(output_path, sid + '_bfc.nii.gz')
    bias_field = os.path.join(output_path, sid + '_bias_field.nii.gz')
    n4_filter = N4BiasFieldCorrection()
    time_0 = time()
    try:
        n4_filter.run_correction(input_path, mask_dilated, bold_bfc, bias_field, method="sitk3D")
        elapsed_time = int(time() - time_0)  
        minutes = elapsed_time // 60
        seconds = elapsed_time - minutes * 60
        print(f'Corrected bold image saved to: {bold_bfc}')
        print(f'Estimated bias field image saved to: {bias_field}')
        print(f"Elapsed time for bias field correction: {minutes} minutes and {seconds} seconds")
    except Exception as e:
        print(f"An error occurred in biad field correction: {e}")
        
    # ------------------ Basic (hierarchical) Motion Correction -------------------
    motion_directory = os.path.join(output_path, 'motion_correction')
    os.makedirs(motion_directory, exist_ok=True)
    _, base = get_filenames(bold_bfc)
    bold_img = nib.load(bold_bfc)
    
    motion_correction = MotionCorrection(
        reference_volume=None,
        registration_method='SimpleITK',
        mask=mask_dilated,
        verbose=True,
        interleave_factor=3,
        hierarchical=False,  
        output_directory=motion_directory)
    
    motion_correction._create_reference(bold_bfc, mask_dilated)
    
    for vol in range(img.shape[-1]):
        volume_data = img[..., vol]
        volume_img = nib.Nifti1Image(volume_data, bold_img.affine, bold_img.header)
        volume_nii = os.path.join(motion_directory, base + f"_vol{vol + 1:03d}.nii.gz")
        nib.save(volume_img, volume_nii)
        motion_correction._volumes.append(volume_nii)
    
    # --------------------- first reconstruction  ------------------------
    
    # ----- two-step slice-to-volume registration-reconstruction ---------
    
    # ---------------------- Outlier Rejection ---------------------------
    
    # --------------------- Nuisance Regression --------------------------
    
    
    
    


    

    
    
    
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    elapsed_time_total = int(time() - time_start)
    minutes = elapsed_time_total // 60
    seconds = elapsed_time_total - minutes * 60
    logger.success('The entire preprocessing has been performed in %dmin %dsec' % (minutes, seconds))
    print("done.")

if __name__ == "__main__":
    main()
    