import os
import numpy as np

HOME_FOLDER = '/Users/athena/Documents/CIRHome/PreprocWorkflow'
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'workspace')

NIFTYREG_PATH = os.path.join(WORKSPACE_FOLDER, 'third-party', 'niftyreg', 'build', 'reg-apps')
ANTS_PATH = '/Users/athena/ants/bin'
FSL_PATH = '/Users/athena/fsl/bin'
AFNI_PATH = '/Users/athena/afni/bin'
LABELS = {
    'white_matter': [120, 121],  
    'csf': [124]
}