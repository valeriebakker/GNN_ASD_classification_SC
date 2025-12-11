#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/subject/dti_1"
  exit 1
fi

if [[ $# -ge 2 ]]; then
  t1_img="$2"
else
  t1_img=""
fi

dti_dir=$1
cd "$dti_dir"

echo "Processing subject from: $dti_dir"

# FSL set up
export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH="$FSLDIR/bin:$PATH"

# Paths to FA MNI-template and AAL-atlas
FA_MNI_TEMPLATE="$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz"
AAL_LABELS="/home/vbakker/atlases/AAL/aal2.nii.gz"


##### PRE-PROCESSING OF DTI-IMAGES #####

# Input b-values and b-vectors
bval_file="dti.bvals"
bvec_file="dti.bvecs"

# Convert DWI to MRtrix format and import FSL gradients
mrconvert dti.nii.gz dwi.mif -fslgrad "$bvec_file" "$bval_file" -force

# Denoise
dwidenoise dwi.mif dwi_denoised.mif -noise noise.mif -force

# Remove Gibbs ringing artefacts
mrdegibbs dwi_denoised.mif dwi_denoised_degibbs.mif -force

# Convert to .nii.gz for eddy and export gradient files
mrconvert dwi_denoised_degibbs.mif dwi_pre_eddy.mif -force
mrconvert dwi_denoised_degibbs.mif dwi_pre_eddy.nii.gz -export_grad_fsl bvecs bvals -force

# Create an index file indicating that all volumes use the same acquisition parameters
nvol=$(mrinfo dwi_pre_eddy.mif -size | awk '{print $4}')
printf "1 %.0s" $(seq 1 $nvol) | sed 's/ $/\n/' > index.txt

# Extract all b0s and average them (needed for motion correction and BET)
dwiextract dwi_pre_eddy.mif - -bzero | mrmath - mean -axis 3 mean_b0_preeddy.mif -force
mrconvert mean_b0_preeddy.mif mean_b0_preeddy.nii.gz -force

# Skull strip the mean b0 using HD-BET and create a brain mask
hd-bet -i mean_b0_preeddy.nii.gz -o mean_b0_preeddy_brain.nii.gz -device cuda:0
fslmaths mean_b0_preeddy_brain.nii.gz -bin mean_b0_preeddy_brain_mask.nii.gz

# Eddy current and motion correction
eddy_cuda11.0 \
  --imain=dwi_pre_eddy.nii.gz \
  --acqp=acqparams.txt --index=index.txt \
  --bvecs=bvecs --bvals=bvals \
  --mask=mean_b0_preeddy_brain_mask.nii.gz \
  --repol \
  --out=dwi_eddy \
  --verbose

## Create a brain mask from the eddy-corrected DWI:

# Convert eddy-corrected DWI back to MRtrix format and use rotated b-vectors
mrconvert dwi_eddy.nii.gz dwi_eddy.mif \
    -fslgrad dwi_eddy.eddy_rotated_bvecs bvals -force

# Extract all b=0 volumes from the eddy-corrected data and average them
dwiextract dwi_eddy.mif - -bzero | mrmath - mean -axis 3 mean_b0_eddy.mif -force

# Convert the averaged b0 to .nii.gz for skull stripping
mrconvert mean_b0_eddy.mif mean_b0_eddy.nii.gz -force

# Skull-strip the eddy-corrected mean b0 using HD-BET
hd-bet -i mean_b0_eddy.nii.gz -o mean_b0_eddy_brain.nii.gz -device cuda:0

# Convert skull-stripped output to a binary mask
fslmaths mean_b0_eddy_brain.nii.gz -bin mean_b0_eddy_brain_mask.nii.gz

# Bias field correction on eddy-corrected data
dwibiascorrect fsl dwi_eddy.mif dwi_biascorr.mif \
  -mask mean_b0_eddy_brain_mask.nii.gz -force

# Intensity normalisation
dwinormalise individual dwi_biascorr.mif mean_b0_eddy_brain_mask.nii.gz dwi_norm.mif -force

# Tensor fitting to extract dti_FA.nii.gz
mrconvert dwi_norm.mif dwi_norm.nii.gz -force
dtifit --data=dwi_norm.nii.gz \
       --out=dti \
       --mask=mean_b0_eddy_brain_mask.nii.gz \
       --bvecs=dwi_eddy.eddy_rotated_bvecs \
       --bvals=bvals


echo "DTI preprocessing finished for $dti_dir"

rm -f \
  dwi.mif dwi_denoised.mif dwi_denoised_degibbs.mif \
  noise.mif dwi_pre_eddy.* mean_b0_preeddy* \
  dwi_eddy.nii.gz dwi_eddy.mif \
  mean_b0_eddy.mif mean_b0_eddy.nii.gz \
  dwi_biascorr.mif dwi_norm.nii.gz \
  dti_L*.nii.gz dti_V*.nii.gz \
  dwi_eddy.eddy_{cnr_maps.nii.gz,outlier_*,rms,movement_rms} \
  2>/dev/null || true   # if files are already removed, no error is given



##### MNI-REGISTRATION #####

# Register AAL2 (MNI) -> subject FA

echo "Registering FA to MNI and warping AAL2 into FA space..."

# FA->MNI (affine + nonlin), then invert
flirt -ref "$FA_MNI_TEMPLATE" -in dti_FA -omat FA_to_MNI_affine.mat
fnirt --ref="$FA_MNI_TEMPLATE" --in=dti_FA --aff=FA_to_MNI_affine.mat --cout=FA_to_MNI_warp --config=FA_2_FMRIB58_1mm
invwarp --ref=dti_FA --warp=FA_to_MNI_warp --out=MNI_to_FA_warp

# Apply AAL to FA space (nearest-neighbour to preserve labels)
applywarp --ref=dti_FA --in="$AAL_LABELS" --warp=MNI_to_FA_warp --out=AAL_in_FA --interp=nn

echo "AAL2 ready for connectome: AAL_in_FA.nii.gz"



### CHECK FOR T1-IMAGE ###

if [[ -n "$t1_img" && -f "$t1_img" ]]; then

  ### T1-REGISTRATION AND WM MASK COMPUTATION ###
  echo "Performing T1 pre-processing"

  # Skull-stripping
  hd-bet -i "$t1_img" -o T1_stripped.nii.gz -device cuda:0

  # Generate WM segmentation for BBR from T1
  5ttgen fsl T1_stripped.nii.gz 5tt_T1.nii.gz -premasked -nocrop -force
  mrconvert 5tt_T1.nii.gz -coord 3 2 T1_wmprob.nii.gz -force
  fslmaths T1_wmprob.nii.gz -thr 0.5 -bin T1_WMseg.nii.gz

  # Initial rigid-body alignment (DWI -> T1)
  flirt -in mean_b0_eddy_brain.nii.gz \
        -ref T1_stripped.nii.gz \
        -dof 6 \
        -cost mutualinfo \
        -omat DWI_to_T1.mat

  # Boundary-based registration (BBR) refinement (DWI -> T1)
  flirt -in mean_b0_eddy_brain.nii.gz \
        -ref T1_stripped.nii.gz \
        -dof 6 \
        -cost bbr \
        -wmseg T1_WMseg.nii.gz \
        -init DWI_to_T1.mat \
        -omat DWI_to_T1_bbr.mat \
        -schedule ${FSLDIR}/etc/flirtsch/bbr.sch

  # Convert FLIRT matrix to MRtrix format
  transformconvert DWI_to_T1_bbr.mat \
                  mean_b0_eddy_brain.nii.gz \
                  T1_stripped.nii.gz \
                  flirt_import DWI_to_T1_bbr-mrtrix.txt -force

  # Apply inverse transform (T1 -> DWI)
  mrtransform T1_stripped.nii.gz T1_DWI_space.nii.gz \
              -linear DWI_to_T1_bbr-mrtrix.txt \
              -inverse \
              -template mean_b0_eddy_brain.nii.gz \
              -force

  # Turn 5TT in DWI space
  mrtransform 5tt_T1.nii.gz 5tt_DWI.nii.gz \
              -linear DWI_to_T1_bbr-mrtrix.txt \
              -inverse \
              -interp nearest \
              -template mean_b0_eddy_brain.nii.gz \
              -force

  # Create GMWM interface
  5tt2gmwmi 5tt_DWI.nii.gz gmwmi_DWI.mif -force


  ### COMPUTE FIBER ORIENTATION DISTRIBUTIONS (FODs) ###

  # Compute response function (ABIDE-II only has one unique b-value next to 0, so single-shell acquisition -> use tournier)
  dwi2response tournier dwi_norm.mif resp_wm.txt -mask mean_b0_eddy_brain_mask.nii.gz -force

  # Single-shell constrained spherical deconvolution (CSD) to compute fiber orientation distributions (FODs)
  dwi2fod csd dwi_norm.mif resp_wm.txt fod_wm.mif -mask mean_b0_eddy_brain_mask.nii.gz -force

  
  ## DETERMINISTIC TRACTOGRAPHY USING ... ###

  echo "Running deterministic tensor tractography..."

  tckgen fod_wm.mif tracks.tck \
    -algorithm sd_stream \
    -act 5tt_DWI.nii.gz \
    -seed_gmwmi gmwmi_DWI.mif \
    -maxlength 250 \
    -angle 45 \
    -select 1000000 \
    -force



### IF NO T1-IMAGE PROVIDED:

else
  echo "No T1 provided; using FA-only WM mask"

  # Create WM-mask by using threshold of FA > 0.2 
  mrthreshold dti_FA.nii.gz -abs 0.2 FA_mask.mif -force
  mrconvert mean_b0_eddy_brain_mask.nii.gz mean_b0_eddy_brain_mask.mif -force
  mrcalc FA_mask.mif mean_b0_eddy_brain_mask.mif -mult wm_mask.mif -force


  ### DETERMINISTIC TRACTOGRAPHY USING FA-THRESHOLDED WM-MASK ###

  echo "Running deterministic tensor tractography..."

  tckgen dwi_norm.mif tracks.tck \
    -algorithm Tensor_Det \
    -seed_image wm_mask.mif \
    -mask wm_mask.mif \
    -maxlength 250 \
    -angle 45 \
    -select 1000000 \
    -force

fi


### COMPUTATION OF STRUCTURAL CONNECTOME ###

tckmap tracks.tck tracks_map.nii.gz -template dti_FA.nii.gz -force

# Connectomes
if [[ -f AAL_in_FA.nii.gz ]]; then
  mrconvert AAL_in_FA.nii.gz AAL_in_FA.mif -force
  mrconvert dti_FA.nii.gz dti_FA.mif -force

  # Streamline count
  tck2connectome tracks.tck AAL_in_FA.mif conn_count.csv \
    -assignment_radial_search 2 \
    -zero_diagonal \
    -force

  tcksample tracks.tck dti_FA.mif fa_per_streamline.txt -stat_tck mean -force

  # FA-weighted mean along edges
  tck2connectome tracks.tck AAL_in_FA.mif conn_FAmean.csv \
    -scale_file fa_per_streamline.txt \
    -stat_edge mean \
    -assignment_radial_search 2 \
    -zero_diagonal \
    -force

  echo "Saved connectomes: conn_count.csv, conn_FAmean.csv"

else
  echo "WARNING: AAL_in_FA.nii.gz not found --> skipping connectome computation."
fi

echo "Finished $dti_dir"

# Remove intermediate files
rm -f \
  AAL_in_FA.mif FA_to_MNI_affine.mat FA_to_MNI_warp* MNI_to_FA_warp* \
  fa_per_streamline.txt rf_wm.txt \
  T1_wmprob.nii.gz T1_WMseg.nii.gz \
  DWI_to_T1.mat DWI_to_T1_init.mat DWI_to_T1_bbr.mat DWI_to_T1_bbr-mrtrix.txt \
  5tt_T1.nii.gz T1_DWI_space.nii.gz T1_stripped.nii.gz \
  2>/dev/null || true   # if files are already removed, no error is given

