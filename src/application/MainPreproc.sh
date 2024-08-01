#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --time=10:00:00
#SBATCH --partition=centos7

module load Python/3.6.8-foss-2019a
module load AFNI/19.1.00-foss-2019a-Python-3.6.8
module load ANTs
module load NiftyReg/20180328-foss-2019a
module load C3D/1.0.0
module load FSL/5.0.11-foss-2019a-Python-3.6.8
. ${FSLDIR}/etc/fslconf/fsl.sh
FUNDIR=/scratch/FetalfMRI/Code

#=================================================
# --sub
#    sub.nii.gz
#    sub_manualmask.nii.gz
#    ManualAlign.txt
#    --Anatomy
#         SEG_##.nii.gz
#         SEG_PAR_##.nii
#         MASKED_##_HR.nii.gz
#==================================================

sub=${subID}
WDIR=/path/to/data/${sub}
echo ${sub}
echo ${WDIR}

python ${FUNDIR}/ReferenceGeneration.py "${sub}" "${WDIR}"
fslmaths ${WDIR}/${sub}_manualmask.nii.gz -kernel sphere 10 -dilD ${WDIR}/${sub}_manualmask_dilD.nii.gz
fslroi ${WDIR}/${sub}_reference.nii.gz ${WDIR}/${sub}_reference_CROPPED.nii.gz `fslstats ${WDIR}/${sub}_manualmask_dilD.nii.gz -w`
nvol=$(fslinfo $WDIR/${sub}.nii.gz | awk '$1 == "dim4" {print $2}')

for i in `seq 1 ${nvol}`
do
var=$var$Data/${sub}/${sub}_manualmask_dilD.nii.gz,
done
var=${var%,}
echo ${var}
N4BiasFieldCorrection --image-dimensionality 4 --input-image ${WDIR}/${sub}.nii.gz --mask-image [$var] --shrink-factor 3 --bspline-fitting [ 300, 3] --output [ ${WDIR}/${sub}_bfc.nii.gz, ${WDIR}/${sub}_bias.nii.gz]
echo "Bias Field Correction Done."

[ ! -d "${WDIR}/MOCO" ] || mkdir -p "${WDIR}/MOCO"
fslsplit ${WDIR}/${sub}_bfc.nii.gz ${WDIR}/MOCO/${sub}_bfc_vol -t

for vol in ${WDIR}/MOCO/${sub}_bfc_vol*
do
tmp=${vol%.nii*}
#fslmaths ${vol} -mul ${WDIR}/${sub}_manualmask_dilD.nii.gz ${tmp}_MASKED.nii.gz
fslroi ${vol} ${tmp}_CROPPED.nii.gz `fslstats ${WDIR}/${sub}_manualmask_dilD.nii.gz -w`
antsRegistration --verbose 1 --dimensionality 3 --float 1 --collapse-output-transforms 1 --output [ ${tmp},${tmp}_Warped.nii.gz,${tmp}_InverseWarped.nii.gz ] --interpolation Linear --use-histogram-matching 1 --winsorize-image-intensities [ 0.005,0.995 ] --initial-moving-transform [ ${WDIR}/${sub}_reference_CROPPED.nii.gz,${tmp}_CROPPED.nii.gz,1 ] --transform Rigid[ 0.01 ] --metric MI[ ${WDIR}/${sub}_reference_CROPPED.nii.gz,${tmp}_CROPPED.nii.gz,1,32,None] --convergence [ 1000x500x250x0,1e-6,10 ] --shrink-factors 1x1x1x1 --smoothing-sigmas 0x0x0x0vox
var1="$var1 ${tmp}_Warped.nii.gz"
done
fslmerge -t ${WDIR}/${sub}_bfc_moco.nii.gz ${var1}
for i in `seq -f "%04g" 0 ${nvol}-1`
do
filename="${WDIR}/MOCO/${sub}_bfc_vol${i}_Warped.nii.gz"
tmp=${filename%_Warp*}
antsRegistration --verbose 1 --dimensionality 3 --float 1 --collapse-output-transforms 1 --output [ ${tmp}_2nd,${tmp}_2ndWarped.nii.gz,${tmp}_2ndInverseWarped.nii.gz ] --interpolation Linear --use-histogram-matching 1 --winsorize-image-intensities [ 0.005,0.995 ] --initial-moving-transform [ ${WDIR}/${sub}_reference_CROPPED.nii.gz,${tmp}_Warped.nii.gz,1 ] --transform Rigid[ 0.01 ] --metric MI[ ${WDIR}/${sub}_reference_CROPPED.nii.gz,${tmp}_Warped.nii.gz,1,32,None] --convergence [ 1000x500x250x0,1e-6,10 ] --shrink-factors 1x1x1x1 --smoothing-sigmas 0x0x0x0vox
var2="$var2 ${tmp}_2ndWarped.nii.gz"
done
fslmerge -t ${WDIR}/${sub}_bfc_mocoANTs.nii.gz ${var2}
[ ! -d "${WDIR}/MOCO/Aladin" ] || mkdir -p "${WDIR}/MOCO/Aladin"
fslsplit ${WDIR}/${sub}_bfc.nii.gz ${WDIR}/MOCO/Aladin/${sub}_bfc_vol -t
for vol in ${WDIR}/MOCO/Aladin/${sub}_bfc_vol*
do
tmp=${vol%.nii*}
fslroi ${vol} ${tmp}_CROPPED.nii.gz `fslstats ${WDIR}/${sub}_manualmask_dilD.nii.gz -w`
#reg_aladin -ref ${WDIR}/${sub}_reference.nii.gz -flo ${tmp}_MASKED.nii.gz -cog -noSym -rmask ${WDIR}/${sub}_manualmask_dilD.nii.gz -res ${tmp}_moco.nii.gz -aff ${tmp}.txt
reg_aladin -ref ${WDIR}/${sub}_reference_CROPPED.nii.gz -flo ${tmp}_CROPPED.nii.gz -affDirect -res ${tmp}_moco.nii.gz -aff ${tmp}.txt
reg_resample -ref ${WDIR}/${sub}_reference_CROPPED.nii.gz -flo ${tmp}_CROPPED.nii.gz -trans ${tmp}.txt -inter 1 -res ${tmp}_mocoLin.nii.gz
var3="$var3 ${tmp}_mocoLin.nii.gz"
done
echo ${var3}
fslmerge -t ${WDIR}/${sub}_bfc_mocoNiftyReg.nii.gz ${var3}
echo "Motion Correction Done."
python ${FUNDIR}/RealignmentEstimates.py "${sub}" "${nvol}" "${WDIR}"
SEG=${WDIR}/Anatomy/SEG_CRL_*
set -- $SEG
c3d ${WDIR}/${sub}_reference.nii.gz $1  -interpolation NearestNeighbor -reslice-itk ${WDIR}/AffineAlign.txt  -o ${WDIR}/${sub}_AffineAlignedSegs.nii.gz
c3d ${WDIR}/${sub}_reference.nii.gz $2  -interpolation NearestNeighbor -reslice-itk ${WDIR}/AffineAlign.txt  -o ${WDIR}/${sub}_AffineAlignedParcels.nii.gz
fslroi ${WDIR}/${sub}_AffineAlignedSegs.nii.gz ${WDIR}/${sub}_AffineAlignedSegs.nii.gz `fslstats ${WDIR}/${sub}_manualmask_dilD.nii.gz -w`
fslroi ${WDIR}/${sub}_AffineAlignedParcels.nii.gz ${WDIR}/${sub}_AffineAlignedParcels.nii.gz `fslstats ${WDIR}/${sub}_manualmask_dilD.nii.gz -w`

registration_types=("ANTs" "NiftyReg" "4DRecon")
for reg_type in "${registration_types[@]}"; do
python Code/RefinedMaskGeneration.py "${sub}" "${WDIR}" "${reg_type}"
3dToutcount -mask ${WDIR}/${sub}_bfc_moco${reg_type}_refinedmask.nii.gz -fraction -polort 3 -legendre ${WDIR}/${sub}_bfc_moco${reg_type}.nii.gz > ${WDIR}/outcount${reg_type}.1D
1deval -a ${WDIR}/outcount${reg_type}.1D -expr 't*step(a-0.03)' | grep -v 'o' > ${WDIR}/outliers${reg_type}.txt
python -c "import nibabel as nib; import numpy as np; data = nib.load('${WDIR}/${sub}_bfc_moco${reg_type}.nii.gz'); img=data.get_fdata(); M=np.genfromtxt('${WDIR}/outliers${reg_type}.txt'); \
idx=np.argwhere(M); new_img = np.delete(img,idx,axis=3); hdr=data.header; hdr['dim'][4] = hdr['dim'][4] - idx.shape[0]; vor = nib.Nifti1Image(new_img,data.affine,hdr); \
nib.save(vor,'${WDIR}/${sub}_bfc_moco${reg_type}_vor.nii.gz')"
3dDespike -NEW -nomask -prefix ${WDIR}/${sub}_bfc_moco${reg_type}_vor_Dspk.nii.gz ${WDIR}/${sub}_bfc_moco${reg_type}_vor.nii.gz
done
[ ! -d "${WDIR}/Regression" ] || mkdir -p "${WDIR}/Regression"
c3d ${WDIR}/${sub}_AffineAlignedParcels.nii.gz -split -foreach -centroid -endfor > ${WDIR}/centers.txt
for reg_type in "${registration_types[@]}"; do
[ ! -d "${WDIR}/Regression/${reg_type}" ] || mkdir -p "${WDIR}/Regression/${reg_type}"
python -c "import scipy.io; th = scipy.io.loadmat('${WDIR}/MotionParam${reg_type}.mat')['cutoff'][0]; print(th[0])" > ${WDIR}/tmpfile.txt
THR=`cat ${WDIR}/tmpfile.txt`
rm ${WDIR}/tmpfile.txt
# we can set it to a fixed value too e.g 1 or 2
python ${FUNDIR}/FilteringRegression.py "${sub}" "${nvol}" "${WDIR}" "${THR}" "${reg_type}" "True" "True"
echo "Nuisance Regression For ${reg_type}-Corrected Images Done."
python ${FUNDIR}/ROIextraction.py "${sub}" "${WDIR}" "${THR}" "${reg_type}" "True" "True"
echo "ROI timeseries Extraction For ${reg_type}-Corrected Images Done."
python ${FUNDIR}/BENCHext.py "${sub}" "${WDIR}" "${THR}" "${reg_type}" "True" "True"
done

















