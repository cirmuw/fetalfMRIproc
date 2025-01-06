import numpy as np
import os
import glob
import multiprocessing as mp
import h5py
import scipy.stats as stats
from scipy import signal
import nibabel as nib
import scipy
import time
import warnings
from utilities.utils import *

import numpy as np

def extract_signal(bold_data, mask_data):
    mask_data = np.asarray(mask_data, dtype=bool)
    signal_data = bold_data[mask_data].copy()
    signal_data = signal.detrend(signal_data, axis=1, type='constant')
    signal_data = signal.detrend(signal_data, axis=1, type='linear')
    return np.mean(signal_data, axis=0)

def get_aCompCor(data, ncomponents=5):
    cov_matrix = np.corrcoef(data.T)
    eigenvalues, topPCs = scipy.sparse.linalg.eigs(cov_matrix, k=ncomponents, which='LM')
    return np.real(topPCs)

def create_phys_regressors(input_nii, mask_nii):
    bold_data = load_nifti_data(input_nii)
    mask_data = load_nifti_data(mask_nii)
    global_signal = extract_signal(bold_data, mask_data)
    wm_signal = extract_signal(bold_data, wm_mask)
    
    

def regression(data, regressors, alpha=0, constant=True):
    """
    closed form equation: betas = (X'X + alpha*I)^(-1) X'y
    Set alpha = 0 for regular OLS.
    Set alpha > 0 for ridge penalty

    PARAMETERS:
        data = observation x feature matrix (e.g., time x regions)
        regressors = observation x feature matrix
        alpha = regularization term. 0 for regular multiple regression. >0 for ridge penalty
        constant = True/False - pad regressors with 1s?

    OUTPUT:
        betas = coefficients X n target variables
        resid = observations X n target variables
    """
    if constant:
        ones = np.ones((regressors.shape[0], 1))
        regressors = np.hstack((ones, regressors))
    X = regressors.copy()

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    LAMBDA = np.identity(X.shape[1]) * alpha
    C_ss_inv = np.linalg.pinv(np.matmul(X.T, X) + LAMBDA)
    betas = np.dot(C_ss_inv, np.matmul(X.T, data))
    resid = data - np.dot(X, betas)

    betas = betas.real
    resid = resid.real

    return betas, resid
