import os
import numpy as np
import nibabel as nib
from utils import run_command

def detect_outliers(sub_id, base_dir, reg_type, refined_mask_path):
    fmri_path = os.path.join(base_dir, f"{sub_id}_bfc_moco{reg_type}.nii.gz")
    outcount_path = os.path.join(base_dir, f"outcount{reg_type}.1D")
    outliers_path = os.path.join(base_dir, f"outliers{reg_type}.txt")

    outcount_command = f"3dToutcount -mask {refined_mask_path} -fraction -polort 3 -legendre {fmri_path} > {outcount_path}"
    run_command(outcount_command)

    eval_command = f"1deval -a {outcount_path} -expr 't*step(a-0.03)' | grep -v 'o' > {outliers_path}"
    run_command(eval_command)

    return outliers_path

def censor_outliers(sub_id, base_dir, reg_type, outliers_path):
    fmri_path = os.path.join(base_dir, f"{sub_id}_bfc_moco{reg_type}.nii.gz")
    vor_path = os.path.join(base_dir, f"{sub_id}_bfc_moco{reg_type}_vor.nii.gz")

    data = nib.load(fmri_path)
    img = data.get_fdata()
    outliers = np.genfromtxt(outliers_path)
    idx = np.argwhere(outliers)
    new_img = np.delete(img, idx, axis=3)

    hdr = data.header
    hdr['dim'][4] = hdr['dim'][4] - idx.shape[0]
    vor_img = nib.Nifti1Image(new_img, data.affine, hdr)
    nib.save(vor_img, vor_path)

    return vor_path

def despike_image(vor_path, despiked_path):
    despike_command = f"3dDespike -NEW -nomask -prefix {despiked_path} {vor_path}"
    run_command(despike_command)