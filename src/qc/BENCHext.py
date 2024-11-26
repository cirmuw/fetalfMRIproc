import os
import sys
import math
import scipy.io
import re
from copy import copy
from scipy import signal
import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist, squareform
from scipy.fftpack import fft, ifft
import scipy.stats


subID=sys.argv[1]
dir=sys.argv[2]
Thr=float(sys.argv[3])
suffix=sys.argv[4]
IsVOR=bool(sys.argv[5])
IsFD=bool(sys.argv[6])

if(IsVOR and IsFD):
    FName = 'VOR_FD'
elif(IsVOR and not IsFD):
    FName = 'VOR_NoFD'
elif(not IsVOR and IsFD):
    FName = 'NoVOR_FD'
else:
    FName = 'NoVOR_NoFD'


parcels_lh = scipy.io.loadmat("/scratch/FetalfMRI/Code/fetalSurface.mat")['parcels_lh']
parcels_rh = scipy.io.loadmat("/scratch/FetalfMRI/Code/fetalSurface.mat")['parcels_rh']

pipeline = ['moco','GSR','2Phys','6HMP','GSR+2Phys+6HMP','24HMP','4GSR+8Phys+24HMP','aCompCor','tCompCor']
regression = ['basis','moco_R1','moco_R2','moco_R3','moco_R4','moco_R5','moco_R6','moco_R7','moco_R8']


def GetDVARS(ts):
    D1 = np.diff(ts, axis=0)
    D1 = np.vstack((np.zeros((1,ts.shape[1])),D1))
    DVARS = np.sqrt(np.mean(np.square(D1), axis=1))
    return DVARS

def dFC(ts,windowlength,Overlap):
    l = ts.shape[1]
    nW = int(np.floor((l-windowlength)/(windowlength-Overlap)))
    m = ts.shape[0]
    cmat = np.zeros((m,m,nW+1))
    for k in range(nW+1):
        data = ts[:,k*(windowlength-Overlap):k*(windowlength-Overlap)+windowlength]
        cc = np.corrcoef(data)
        cc = cc - np.diag(np.diag(cc))
        cmat[:,:,k] = cc
    return cmat

def dFD(FD,windowlength,Overlap):
    l = FD.shape[0]
    nW = int(np.floor((l-windowlength)/(windowlength-Overlap)))
    mm1 = np.zeros((nW+1,1))
    for k in range(nW+1):
        mm1[k] = np.mean(FD[k*(windowlength-Overlap):k*(windowlength-Overlap)+windowlength,0])
    return mm1

def qcfc_dist(dfc_vec,dfd,ROIDistVec):
    FcFdCorr = np.zeros((dfc_vec.shape[1],1))
    FcFdPval = np.zeros((dfc_vec.shape[1],1))
    ShuffledCorr = np.zeros((dfc_vec.shape[1],1000))
    f = fft(dfd)
    r = np.random.rand(f.shape[0],1000)
    ranp = np.angle(fft(r,axis=0))
    tmp = np.abs(f)*np.sqrt(-1+0j)*ranp
    shuffled_fd = np.real(ifft(tmp,axis=0)) 
    for jj in range(dfc_vec.shape[1]):
        corr = scipy.stats.pearsonr(dfc_vec[:,jj],dfd[:,0])
        corr = np.array(corr)
        FcFdCorr[jj,0] = corr[0]
        FcFdPval[jj,0] = corr[1]
        ShuffledCorr[jj,:] = np.corrcoef(shuffled_fd.T,dfc_vec[:,jj].T)[:-1,-1]
    ind = np.argwhere(FcFdPval>0.05)[:,0]
    SigFcFdCorr = np.delete(FcFdCorr, ind, axis=0)
    nbOfSigCorr = SigFcFdCorr.shape[0]
    mdlCoef = np.polyfit(ROIDistVec,FcFdCorr[:,0],1)
    p = np.poly1d(mdlCoef)
    return nbOfSigCorr, FcFdCorr, FcFdPval, ShuffledCorr, mdlCoef, p

brain_mask = nib.load(dir + subID + "_AffineAlignedParcels.nii.gz")
brain_seg = nib.load(dir + subID + "_AffineAlignedSegs.nii.gz")
parc = brain_mask.get_fdata()
parc = np.reshape(parc,(-1,1)).T
seg = brain_seg.get_fdata()
seg = np.reshape(seg,(-1,1)).T

all_label_indices = np.unique(parc)
label_indices_lh = np.unique(parcels_lh)
label_indices_rh = np.unique(parcels_rh)
label_indices_lr = np.zeros((98,1))
label_indices_lr[0:98:2,0] = label_indices_lh
label_indices_lr[1:98:2,0] = label_indices_rh
cortical_id = np.hstack((np.arange(1,37),[39,40],np.arange(43,71),np.arange(79,91)))

M = np.genfromtxt(dir + "outliers" + suffix + ".txt")
fdRMS = scipy.io.loadmat(dir + "MotionParam" + suffix + ".mat")['fd']
ind1 = np.argwhere(M)
cs_fdRMS = np.delete(fdRMS,ind1,axis=1) #intensity-based
#ind2 = np.argwhere(cs_fdRMS>Thr)[:,1]   #the indices of FD-based volumes in addition to vor-based
ind2 = np.argwhere(fdRMS>Thr)[:,1]
fdcs_fdRMS = np.delete(fdRMS,ind2,axis=1)
#we remove these volumes from timeseries (_vor already) in order to estimate regression coefficients solely based on low-contaminated volumes
#for backup: both outlier volumes and fdRMS>1 are marked
#ind3 = np.argwhere(fdRMS>Thr)[:,1]  #only FD-based indices
ind3 = np.argwhere(cs_fdRMS>Thr)[:,1]
new_fdRMS = np.delete(cs_fdRMS, ind3, axis=1)  #both intensity and FD-based

scrubmask = fdRMS > Thr
scrubmask[0,ind1] = True

text_file = open(dir + "centers.txt", "r")
lines = text_file.read().split('\n')
center_mm = np.empty((0,3))
for mystr in lines[3::2]:
    mm = np.asarray(re.findall(r"[-+]?\d*\.\d+|\d+", mystr))
    mm = mm[..., np.newaxis].T
    center_mm = np.append(center_mm, mm, axis=0)
full_centers = np.zeros((124,3))
for jj in range(1,len(all_label_indices)):
    ii = all_label_indices[jj]
    if ii != 125:
        full_centers[int(ii)-1,:] = center_mm[jj-1,:]
d = pdist(full_centers)
Dist = squareform(d)
scipy.io.savemat(dir + "distance.mat", {"Dist": Dist, "centroid":full_centers})

WL = 16
OL = 12
os.makedirs(dir + "BENCH_" + str(WL)+str(OL)+"/" + FName + "/"+ suffix + "/Thr="+format(Thr, '.2f')+"/")
for qq in range(9):
    print(pipeline[qq])
    if (qq == 0 and IsVOR):
        rsData = nib.load(dir + subID + "_bfc_moco" + suffix + "_vor_Dspk.nii.gz")
    elif (qq == 0 and not IsVOR):
        rsData = nib.load(dir + subID + "_bfc_moco" + suffix + "_Dspk.nii.gz")
    else:
        rsData = nib.load(dir + "Regression/"+ FName + "/" + suffix + "/Thr="+format(Thr, '.2f')+"/" + regression[qq] + ".nii.gz")
    img = rsData.get_fdata()
    ts2 = np.reshape(img,(-1,1)).T
    idx2 = np.argwhere(np.isnan(ts2))
    ts2[idx2[:,1]]=0
    img = np.reshape(ts2,img.shape)
    dim = img.shape
    ts = np.reshape(img,(-1,dim[3])).T
    if (qq == 0):
        if(IsFD and IsVOR):
            ts_scrubbed = np.delete(ts,ind3,axis=0)
        elif(IsFD and not IsVOR):
            ts_scrubbed = np.delete(ts,ind2,axis=0)
        else:
            ts_scrubbed = copy(ts)
    else:
        ts_scrubbed = copy(ts)
    #Define mask2nd not based on the nan_ids but instead based on all-zero voxels
    zid = np.argwhere(np.any((ts_scrubbed==0),axis=0))
    mask2nd = np.zeros((1,ts_scrubbed.shape[1]))
    mask2nd[0,zid] = 1
    mask2nd = np.array(mask2nd, dtype=bool)
    parc2 = copy(parc)
    parc2 = np.array(parc2,dtype=bool)
    mask = parc2 & ~mask2nd
    ts_scrubbed = ts_scrubbed[:,mask[0,:]]
    

    DVARS = GetDVARS(ts_scrubbed)
    if (IsFD and IsVOR):
        fd_dvars_corr = scipy.stats.pearsonr(DVARS[1:],new_fdRMS[0,1:])
        mfd = dFD(new_fdRMS.T,WL,OL)
    elif (IsFD and not IsVOR):
        fd_dvars_corr = scipy.stats.pearsonr(DVARS[1:],fdcs_fdRMS[0,1:])
        mfd = dFD(fdcs_fdRMS.T,WL,OL)
    elif(not IsFD and IsVOR):
        fd_dvars_corr = scipy.stats.pearsonr(DVARS[1:],cs_fdRMS[0,1:])
        mfd = dFD(cs_fdRMS.T,WL,OL)
    else:
        fd_dvars_corr = scipy.stats.pearsonr(DVARS[1:],fdRMS[0,1:])
        mfd = dFD(fdRMS.T,WL,OL)
    fd_dvars_corr = np.array(fd_dvars_corr)

    ERODED_ROIs = scipy.io.loadmat(dir + "Regression/"+ FName + "/" + suffix + "/Thr="+format(Thr, '.2f')+"/roi_"+ regression[qq] + ".mat")['ERODED_ROIs']
    cortical_roi = ERODED_ROIs[cortical_id-1,:]
    sfc = np.corrcoef(cortical_roi)
    FChist = sfc[np.triu_indices(cortical_id.shape[0], k = 1)]
    MedVal = np.median(FChist)
    #if IsFD:
    #    mfd = dFD(new_fdRMS.T,16,12)
    #else:
    #    mfd = dFD(cs_fdRMS.T,16,12)
    cmat = dFC(cortical_roi,WL,OL)
    nc = int(cortical_id.shape[0]*(cortical_id.shape[0]-1)/2)
    FCvec = np.zeros((cmat.shape[2],nc))
    for ii in range(cmat.shape[2]):
        tmp = cmat[:,:,ii]
        FCvec[ii,:] = tmp[np.triu_indices(cortical_id.shape[0], k = 1)]
    cortical_center = full_centers[cortical_id-1,:]
    crt_d = pdist(cortical_center)
    CrtDist = squareform(crt_d)
    DistVec = CrtDist[np.triu_indices(cortical_id.shape[0], k = 1)]
    nbOfSigCorr, FcFdCorr, FcFdPval, ShuffledCorr, mdlCoef, p = qcfc_dist(FCvec,mfd,DistVec)
    with open(dir + "BENCH_" + str(WL)+str(OL)+"/"+ FName + "/" + suffix + "/Thr="+format(Thr, '.2f')+"/bench_" + regression[qq] + ".mat", 'wb') as f:
        scipy.io.savemat(f, {'nbOfSigCorr': nbOfSigCorr})
        scipy.io.savemat(f, {'FcFdCorr': FcFdCorr})
        scipy.io.savemat(f, {'FcFdPval': FcFdPval})
        scipy.io.savemat(f, {'ShuffledCorr': ShuffledCorr})
        scipy.io.savemat(f, {'mdlCoef': mdlCoef})
        scipy.io.savemat(f, {'FcFdDist': p})
        scipy.io.savemat(f, {'MedVal': MedVal})
        scipy.io.savemat(f, {'sfc': sfc})
        scipy.io.savemat(f, {'FChist': FChist})
        scipy.io.savemat(f, {'fd_dvars_corr': fd_dvars_corr})
        scipy.io.savemat(f, {'DVARS': DVARS})
        scipy.io.savemat(f, {'fdRMS_scrubbed': new_fdRMS})
        scipy.io.savemat(f, {'ts_scrubbed': ts_scrubbed})
        scipy.io.savemat(f, {'DistVec': DistVec})









