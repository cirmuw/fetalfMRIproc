##
# \file run_preprocessing_workflow.py
# \brief      Script to preprocess raw in-utero fmri images (4d) and derive 
#             clean timeseries
#
# \author     Athena Taymourtash (athena.taymourtash@meduniwien.ac.at)
# \date       August 2024
#
import os
import sys
import numpy as np
from time import time
from loguru import logger
from argparse import ArgumentParser
from src.utilities.utils import *
from src.utilities.definitions import LABELS
from src.utilities.bias_field_correction import N4BiasFieldCorrection
from src.utilities.base_motion_correction import MotionCorrection
from src.utilities.volume_outlier_detection import OutlierDetection
from src.utilities.spike_rejection import SpikeRejection
from src.utilities.segmentation_propagation import SegmentationPropagation
from src.utilities.nuisance_regressor import NuisanceRegressor
from src.utilities.regression_model import RegressionModel
from src.utilities.temporal_filtering import TemporalFiltering


def main():
    
    
    np.set_printoptions(precision=3)
    
    parser = ArgumentParser(description="Fetal rs-fMRI workflow", epilog='\n')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--input-mask', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--T2-to-bold', required=True, type=str)
    parser.add_argument('--is-inverse', required=False, type=bool, default=False)
    parser.add_argument('--segmentation', required=True, type=str)
    parser.add_argument('--parcellation', required=False, type=str)
    parser.add_argument('--registration-method', required=False, type=str, default="SimpleITK")
    parser.add_argument('--registration-type', required=False, type=str, default='rigid')
    parser.add_argument('--n4-method', required=False, type=str, default='sitk3D')
    parser.add_argument('--despiking-method', required=False, type=str, default='Lfit')
    parser.add_argument('--vout-method', required=False, type=str, default='pyToutcount')
    parser.add_argument('--regression-model', required=False, type=str, default='24HMP+8Phys+4GSR')
    parser.add_argument('--num_components', required=False, type=int, default=5)
    parser.add_argument('--temporal-filter', required=False, type=str, default='dct')
    parser.add_argument('--dummy-frames', required=False, type=int, default=0)
    parser.add_argument('--dilation-radius', required=False, type=int, default=5)
    parser.add_argument('--interleave-factor', required=False, type=int, default=3)
    
    
    
    args = parser.parse_args()
    
    input_bold = args.input
    output_path = args.output
    input_mask = args.input_mask
    input_seg = args.segmentation
    if args.parcellation:
        input_parcels = args.parcellation
    transformation = args.T2_to_bold
    is_inverse = args.is_inverse
    num_components = args.num_components
    
    logger.info('Input 4D fMRI: %s' % input_bold)
    logger.info('Input 3D brain mask: %s' % input_mask)
    logger.info('Output folder: %s' % output_path)
    
    interleave_factor = args.interleave_factor
    n4_method = args.n4_method
    registration_method = args.registration_method
    registration_type = args.registration_type
    despiking_method = args.despiking_method
    vout_method = args.vout_method
    reg_model_name = args.regression_model
    temporal_filter_type = args.temporal_filter
    
    sid = get_subject_id(input_bold)
    tr = get_repetition_time(input_bold)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    # -------------------------- Read Data --------------------------
    img = load_nifti_data(input_bold)[0]
    if args.dummy_frames:
        img = del_dummy_scans(img, args.dummy_frames)
    
    num_vol = get_num_vols(img)
    
    dilation_radius = args.dilation_radius
    mask_dilated = dilate_mask(input_mask, os.path.join(output_path, sid + '_mask_dilated.nii.gz'), dilation_radius)
    
    # ---------------------- Bias Field Correction -------------------
    bold_bf_corrected = os.path.join(output_path, sid + '_bfc.nii.gz')
    bias_field_estimate = os.path.join(output_path, sid + '_bias_field.nii.gz')
    n4_filter = N4BiasFieldCorrection()
    time_0 = time()
    try:
        n4_filter.run_correction(input_bold, mask_dilated, bold_bf_corrected, bias_field_estimate, method=n4_method)
        elapsed_time = int(time() - time_0)  
        minutes = elapsed_time // 60
        seconds = elapsed_time - minutes * 60
        print(f'Corrected bold image saved to: {bold_bf_corrected}')
        print(f'Estimated bias field image saved to: {bias_field_estimate}')
        print(f"Elapsed time for bias field correction: {minutes} minutes and {seconds} seconds")
    except Exception as e:
        print(f"An error occurred in biad field correction: {e}")
        
    # ------------------ Basic (hierarchical) Motion Correction -------------------
    motion_directory = os.path.join(output_path, 'motion_correction')
    os.makedirs(motion_directory, exist_ok=True)
    
    _, base = get_filenames(bold_bf_corrected)
    bold_mc = os.path.join(output_path, base + '_mc.nii.gz')
    
    img_bfc, affine_bfc, header_bfc = load_nifti_data(bold_bf_corrected)
    
    motion_correction = MotionCorrection(
        reference_volume=None,
        motion_corrected_nii=bold_mc,
        registration_method=registration_method,
        registration_type=registration_type,
        mask=mask_dilated,
        verbose=True,
        interleave_factor=interleave_factor,
        repetition_time=tr,
        hierarchical=False,  
        motion_directory=motion_directory)
    
    motion_correction._create_reference(bold_bf_corrected, mask_dilated)
    
    for vol in range(img_bfc.shape[-1]):
        volume_data = img_bfc[..., vol]
        volume_img = nib.Nifti1Image(volume_data, affine_bfc, header_bfc)
        volume_nii = os.path.join(motion_directory, base + f"_vol{vol:03d}.nii.gz")
        nib.save(volume_img, volume_nii)
        motion_correction._volumes.append(volume_nii)
        
    motion_correction.run()
    
    # ----- two-step slice-to-volume registration-reconstruction -----------------
    # Please see: "Spatio-temporal motion correction and iterative reconstruction 
    #              of in-utero fetal fMRI"for robust motion correction", MICCAI 2022  
    #
    # ------------- Spike Rejection & Outlier Detection --------------------------
    spike_rejector = SpikeRejection(bold_mc, mask_dilated, method=despiking_method)
    bold_dspk, spikes_mask = spike_rejector.despike()
    
    outlier_detector = OutlierDetection(bold_dspk, mask_dilated, method=vout_method, normalize=True)
    outlier_indices, volume_outliers = outlier_detector.run() 
    
    # ----------- Segmentation/Parcellation Preparation  -----------------
    segmentation_propagator = SegmentationPropagation(transformation, bold_dspk, is_inverse)
    seg_resampled = segmentation_propagator._propagate_labels(input_seg)
    if args.parcellation:
        parcels_resampled = segmentation_propagator._propagate_labels(input_parcels)
    
    # --------------------- Nuisance Regression --------------------------
    nuisance_regressor = NuisanceRegressor(bold_dspk, 
                                       mask_dilated, 
                                       seg_resampled,
                                       motion_directory,  
                                       volume_outliers,
                                       num_components,
                                       wm_labels=LABELS['white_matter'], 
                                       csf_labels=LABELS['csf'],
                                       )

    regressors = nuisance_regressor.x

    regression_model = RegressionModel(regressors, reg_model_name, alpha=0.1, constant=True)

    _, residuals = regression_model._regression_core(bold_dspk)

    _, base = get_filenames(bold_dspk)
    regressed_bold = os.path.join(output_path, base + '_regressed.nii.gz')
    regression_model.save_residual_image(residuals, regressed_bold)
    
    # --------------------- Temporal filtering ---------------------------
    temporal_filter = TemporalFiltering(regressed_bold, tr, lowcut=0.001, highcut=0.01, filter_type=temporal_filter_type, order=5)
    bold_filtered = temporal_filter.apply_filter()
    
    # --------------------------------------------------------------------
    elapsed_time_total = int(time() - time_0)
    minutes = elapsed_time_total // 60
    seconds = elapsed_time_total - minutes * 60
    logger.success('The entire preprocessing has been performed in %dmin %dsec' % (minutes, seconds))
    print("done.")

if __name__ == "__main__":
    main()
    