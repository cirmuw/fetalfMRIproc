import os
import numpy as np
import nibabel as nib

def generate_reference(subID, dir1):

    img_path = os.path.join(dir1, f"{subID}.nii.gz")
    data = nib.load(img_path)
    img_data = data.get_fdata()
       
    n_x, n_y, n_z, n_t = img_data.shape
    nb_vox = n_x * n_y * n_z
    mse = np.zeros(n_t)
     
    for i in range(n_t):
        f1 = img_data[..., i]
        mse_sum = np.sum([np.mean((f1 - img_data[..., j])**2) for j in range(n_t)])
        mse[i] = mse_sum / n_t
    
    
    thr = np.quantile(mse, 0.1)
    valid_indices = np.where(mse <= thr)[0]
    new_img = img_data[..., valid_indices]
      
    ave_img = np.mean(new_img, axis=3)
    ave_header = data.header.copy()
    ave_header['dim'][0] = 3
    ave_header['dim'][4] = 1
    reference = nib.Nifti1Image(ave_img, data.affine, ave_header)
    reference_path = os.path.join(dir1, f"{subID}_reference.nii.gz")
    nib.save(reference, reference_path)
    print("Image Reference Done.")
