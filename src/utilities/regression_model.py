import numpy as np
import nibabel as nib

class RegressionModel:
    def __init__(self, regressors, model_name="24pXaCompCorXVolterra", alpha=0, constant=True):
        self.regressors = regressors
        self.model_name = model_name
        self.alpha = alpha
        self.constant = constant
        self.model_config = self._get_model_config()
        
    def _get_model_config(self):
        model_configs = {
            "GSR": [("global_signal", False, False)],
            "2Phys": [("wm_signal", False, False), ("csf_signal", False, False)],
            "6HMP": [("motion_params", False, False)],
            "6HMP+2Phys": [("wm_signal", False, False), ("csf_signal", False, False),("motion_params", False, False)],
            "6HMP+2Phys+GSR": [("global_signal", False, False),("wm_signal", False, False), ("csf_signal", False, False),("motion_params", False, False)],
            "24HMP": [("motion_params", True, True)],
            "24HMP+8Phys": [("motion_params", True, True),("wm_signal", True, True),("csf_signal", True, True)],
            "24HMP+8Phys+4GSR": [("motion_params", True, True),("wm_signal", True, True),("csf_signal", True, True),("global_signal", True, True)],
            "aCompCor": [("acompcor", False, False)],
            "tCompCor": [("tcompcor", False, False)],
            "24HMP+aCompCor": [("acompcor", True, False),("motion_params", True, True)],
            "12HMP+aCompCor": [("acompcor", False, False),("motion_params", True, False)]
            
        }
        if self.model_name not in model_configs:
            raise ValueError(f"Model '{self.model_name}' is not recognized.")
        
        return model_configs['self.model_name']
    
    def _generate_regressor_mat(self):
        X = []
        for regressor_name, derivatives, quadratics in self.model_config:
            self._add_regressor(X, regressor_name, derivatives, quadratics)

        if self.constant:
            ones = np.ones((len(X[0]), 1)) if X else np.ones((self.regressors['motion_params'].shape[0], 1))
            X = [ones] + X

        X_matrix = np.column_stack(X) if X else None

        return X_matrix
    
    def _add_regressor(self, X, regressor_name, derivatives=False, quadratics=False):
        if regressor_name not in self.regressors:
            return

        reg = self.regressors[regressor_name]
        if reg.ndim == 1:
            reg = reg[:, np.newaxis]
        X.append(reg)

        if derivatives:
            reg_deriv = np.gradient(reg, axis=0)
            X.append(reg_deriv)
        
        if quadratics:
            reg_squared = reg ** 2
            X.append(reg_squared)
            if derivatives:
                reg_deriv_squared = reg_deriv ** 2
                X.append(reg_deriv_squared)
        
    def _apply_regression(self, bold_data, mask_img, model):
        regressors = self._generate_regressor_mat()
        if self.constant:
            ones = np.ones((regressors.shape[0], 1))
            regressors = np.hstack((ones, regressors))
        X = regressors.copy()
        if X is None:
            raise ValueError("Regressor matrix is None. Please check the regressor configuration.")

        n_voxels = np.prod(bold_data.shape[:-1])
        bold_2d = bold_data.reshape(n_voxels, bold_data.shape[-1]).T

        # Apply regression
        LAMBDA = np.identity(X.shape[1]) * self.alpha
        C_ss_inv = np.linalg.pinv(np.matmul(X.T, X) + LAMBDA)
        betas = np.dot(C_ss_inv, np.matmul(X.T, bold_2d))
        resid_2d = bold_2d - np.dot(X, betas)

        # Reshape the residuals back to the original 4D shape
        resid = resid_2d.T.reshape(bold_data.shape)

        betas = betas.real
        resid = resid.real

        return betas, resid
    
    def save_residual_image(self, residuals, affine, header, output_path):
        residual_img = nib.Nifti1Image(residuals, affine=affine, header=header)
        nib.save(residual_img, output_path)
        print(f"Residual image saved to {output_path}")
        