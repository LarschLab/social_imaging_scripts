#!/usr/bin/env bash

ANTSPATH="/raven/u/${USER}/ANTs/antsInstallExample/install/bin"
export ANTSPATH="/raven/u/${USER}/ANTs/antsInstallExample/install/bin"
ANTSBIN="/raven/u/${USER}/ANTs/antsInstallExample/install/bin"
WALL_TIME="24:00:00"
#PARTITION="general"

SUFFIX="HCR_GCaMP6s_stack.nrrd"
SUFFIX_POINTS="2P_GCaMP6s_stack"
SUFFIX_HCRTOREF="gcamp_to_avg"
SUFFIX_FRAME="norot.nrrd"

FILETYPE=".nrrd"
OUTPATH="/raven/u/${USER}/data/fixed/"

TARG=$1
PARTITION=$2 #general, fast, ... specifies what kind of cluster node, gpu, cpu etc
PREFIX=$3 # select a subset of files from folder

POINTPATH_THREE="/raven/u/${USER}/reg/xyz_three"
OUTPATH_THREE="/raven/u/${USER}/reg/moving_three"

POINTPATH_ONE="/raven/u/${USER}/reg/xyz_one"
OUTPATH_ONE="/raven/u/${USER}/reg/moving_one"

BRAINSPATH="/raven/u/${USER}/data/moving/${PREFIX}"

case "$TARG" in

    "align_14dpf")
        for TEMPLATE in /raven/u/${USER}/reg/moving_two/${PREFIX}; do

        REF="/raven/u/${USER}/reg/fixed/14dpf_AVG_H2BGCaMP6s.nrrd"

        MASK_T="/raven/u/${USER}/reg/moving_one/7dpf_mask.nrrd"
        MASK_REF="/raven/u/${USER}/reg/fixed/7dpf_mask_ref.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}},${TEMPLATE:: -${#FILETYPE}}_aligned.tif] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.05,0.95] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.05,6,0.2] \
        -m CC[${TEMPLATE},${REF},1,4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox

        ${ANTSBIN}/antsApplyTransforms \
        -d 3 --verbose 1 \
        -r ${REF} \
        -i ${TEMPLATE} \
        -o ${TEMPLATE%.*}_aligned_ref.tif \
        -t [${TEMPLATE%.*}0GenericAffine.mat , 1] \
        -t ${TEMPLATE%.*}1InverseWarp.nii.gz"

        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-7dpf --wrap="${ANTSCALL}" #--mem=512000
        done
        ;;

    "align_7dpf")
        for TEMPLATE in /raven/u/${USER}/reg/moving_one/${PREFIX}; do

        REF="/raven/u/${USER}/reg/fixed/7dpf_AVG_H2BGCaMP6s_atlas.nrrd"

        MASK_T="/raven/u/${USER}/reg/moving_one/7dpf_mask.nrrd"
        MASK_REF="/raven/u/${USER}/reg/fixed/7dpf_mask_ref.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$MASK_T"
        echo "$MASK_REF"
#hallo
        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}, ${TEMPLATE:: -${#FILETYPE}}_aligned.tif] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.05,0.95] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.2,6,0.1] \
        -m CC[${TEMPLATE},${REF},1,4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox \

        ${ANTSBIN}/antsApplyTransforms -d 3 --verbose 1 \
        -r ${REF} \
        -i ${TEMPLATE} \
        -o ${TEMPLATE%.*}_aligned_ref.nrrd \
        -t [${TEMPLATE%.*}0GenericAffine.mat, 1]\
        -t ${TEMPLATE%.*}1InverseWarp.nii.gz"

        sbatch --partition="$PARTITION" -N 1 -n 1 -c 72 -t "$WALL_TIME" -J ANTs-7dpf --wrap="${ANTSCALL}" --mem=256000
        done
        ;;

    "align_21dpf")
        for TEMPLATE in /raven/u/${USER}/data/moving/${PREFIX}; do

        #REF="/raven/u/${USER}/data/fixed/21dpf_AVG_H2BGCaMP6s.nrrd"
        #REF="/raven/u/${USER}/data/fixed/20230316_F1_Zstack_00001_00001preProcessed_adjust_b.nrrd"
        #REF="/raven/u/${USER}/data/fixed/20230316_F2_Zstack_00001_00001preProcessed_adjust_b.nrrd"
        REF="/raven/u/${USER}/data/fixed/20230316_F3_Zstack_00001_00001preProcessed_adjust_b.nrrd"

        MASK_T="/raven/u/${USER}/data/moving/21dpf_mask.nrrd"
        MASK_REF="/raven/u/${USER}/data/fixed/DTmask.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        # originally SyN [0.05, 6, 0.2], worked in half the cases
        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}},${TEMPLATE:: -${#FILETYPE}}_aligned.tif] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0, 100] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.25,6,0.1] \
        -m CC[${TEMPLATE},${REF},1,4] \
        -c [200x200x200x200x10,1e-7,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox

        ${ANTSBIN}/antsApplyTransforms \
        -d 3 --verbose 1 \
        -r ${REF} \
        -i ${TEMPLATE} \
        -o ${TEMPLATE%.*}_aligned_ref.tif \
        -t ${TEMPLATE%.*}1Warp.nii.gz \
        -t ${TEMPLATE%.*}0GenericAffine.mat"

        sbatch --partition="$PARTITION" -N 1 -n 1 -c 72 -t "$WALL_TIME" -J ANTs-21dpf --wrap="${ANTSCALL}" --mem=256000
        done
        ;;

    "align_hcr_hires")
        for T1 in /raven/u/${USER}/reg/moving/*${SUFFIX}; do

        R1=${T1/moving/fixed}
        R1=${R1/HCR/2P}

        TEMPLATE=${T1}
        REF=${R1}
        echo "$TEMPLATE"
        echo "$REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}},${TEMPLATE:: -${#FILETYPE}}_aligned_syn_01_6_0_10it.tiff] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.05,0.95] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE},${REF},1,32,Regular,0.25] \
        -c [200x200x200x0,1e-8,10] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.1,6,0.0] \
        -m CC[${TEMPLATE},${REF},1,4] \
        -c [200x200x200x200x10,1e-7,10] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox"
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}" #--mem=512000
        done
        ;;

    "align_juv_hcr")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        MASK_T=${TEMPLATE/_masked}
        C2_T=${TEMPLATE/C1/C2}

        MASK_REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd_mask.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd.nrrd"
        C2_REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd_cort.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$C2_T"
        echo "$C2_REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_mc2_,${TEMPLATE:: -${#FILETYPE}}_aligned_mc2.tiff] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.01,0.99] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 0.7, 32, Regular, 0.25] \
        -m MI[${C2_T}, ${C2_REF}, 0.3, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 0.7, 32, Regular, 0.25] \
        -m MI[${C2_T}, ${C2_REF}, 0.3, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.2,6,0] \
        -m CC[${TEMPLATE}, ${REF}, 0.3, 4] \
        -m CC[${C2_T}, ${C2_REF}, 0.7, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox

        ${ANTSBIN}/antsApplyTransforms \
        -d 3 --verbose 1 \
        -r ${REF} \
        -i ${TEMPLATE} \
        -o ${TEMPLATE%.*}_aligned_r_mc2.nrrd \
        -t [${TEMPLATE%.*}_mc2_0GenericAffine.mat , 1] \
        -t ${TEMPLATE%.*}_mc2_1InverseWarp.nii.gz
        "

        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        done
        ;;

    "align_juv_hcr_ventral")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        MASK_T=${TEMPLATE/_masked}
        C2_T=${TEMPLATE/C1/C2}

        MASK_REF="/raven/u/${USER}/reg/fixed/jmask.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v.nrrd"
        C2_REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v_cort.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$C2_T"
        echo "$C2_REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_mc2_,${TEMPLATE:: -${#FILETYPE}}_aligned_mc2.tiff] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.01,0.99] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 0.7, 32, Regular, 0.25] \
        -m MI[${C2_T}, ${C2_REF}, 0.3, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 0.7, 32, Regular, 0.25] \
        -m MI[${C2_T}, ${C2_REF}, 0.3, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.2,6,0] \
        -m CC[${TEMPLATE}, ${REF}, 0.3, 4] \
        -m CC[${C2_T}, ${C2_REF}, 0.7, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox

        ${ANTSBIN}/antsApplyTransforms \
        -d 3 --verbose 1 \
        -r ${REF} \
        -i ${TEMPLATE} \
        -o ${TEMPLATE%.*}_aligned_r_mc2.nrrd \
        -t [${TEMPLATE%.*}_mc2_0GenericAffine.mat , 1] \
        -t ${TEMPLATE%.*}_mc2_1InverseWarp.nii.gz
        "

        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        done
        ;;

    "align_juv_hcr_vd")

        MASK_T="/raven/u/${USER}/reg/fixed/34343" # juvenile_hcr_ref_dv_d_cort_mask.nrrd" # juvenile_hcr_ref_dv_dmask.nrrd"
        TEMPLATE="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_d.nrrd"
        TEMPLATE="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_d_cort.nrrd"

        MASK_REF="/raven/u/${USER}/reg/fixed/34343" # juvenile_hcr_ref_dv_v_cort_mask.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v_cort.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_try3_,${TEMPLATE:: -${#FILETYPE}}_aligned_try3.tif] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.05,0.95] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.2,6,0.1] \
        -m CC[${TEMPLATE}, ${REF}, 1., 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"

        MASK_REF="/raven/u/${USER}/reg/fixed/34343" # juvenile_hcr_ref_dv_d_cort_mask.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_d.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_d_cort.nrrd"

        MASK_T="/raven/u/${USER}/reg/fixed/34343" # juvenile_hcr_ref_dv_v_cort_mask.nrrd"
        TEMPLATE="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v.nrrd"
        TEMPLATE="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v_cort.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_inv_try3,${TEMPLATE:: -${#FILETYPE}}_aligned_inv_try3.tif] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.05,0.95] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.2,6,0.1] \
        -m CC[${TEMPLATE}, ${REF}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        ;;

    "align_hcr_2p")

        TEMPLATE="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_mcg_bspline.nrrd"
        MASK_REF="/raven/u/${USER}/reg/fixed/34343" # 21dpf_AVG_H2BGCaMP6s_gamma04_mask.nrrd"
        REF="/raven/u/${USER}/reg/fixed/21dpf_AVG_H2BGCaMP6s_gamma04_gauss3.nrrd"
        MASK_T="/raven/u/${USER}/reg/fixed/34343" # juvenile_hcr_ref_vd_gamma04_mask_zres2.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_gauss_try9,${TEMPLATE:: -${#FILETYPE}}_aligned_gauss_try9.tiff] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.02,0.98] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.1,6,0.2] \
        -m CC[${TEMPLATE}, ${REF}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${REF:: -${#FILETYPE}}_gauss_try9_inv,${REF:: -${#FILETYPE}}_aligned_gauss_try9_inv.tiff] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.02,0.98] \
        --use-histogram-matching 1 \
        -r [${REF},${TEMPLATE},1] \
        -t rigid[0.1] \
        -m MI[${REF}, ${TEMPLATE}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_REF}, ${MASK_T}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${REF}, ${TEMPLATE}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_REF}, ${MASK_T}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.1,6,0.2] \
        -m CC[${REF}, ${TEMPLATE}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_REF}, ${MASK_T}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        ;;

    "align_hcr_2p_syn")

        TEMPLATE="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_mcg_bspline.nrrd"
        REF="/raven/u/${USER}/reg/fixed/21dpf_AVG_H2BGCaMP6s.nrrd"

        echo "$TEMPLATE"
        echo "$REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_dv_try5,${TEMPLATE:: -${#FILETYPE}}_dv_try5_aligned.tiff] \
        --interpolation WelchWindowedSinc \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t SyN[0.2,6,0.1] \
        -m CC[${TEMPLATE}, ${REF}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${REF:: -${#FILETYPE}}_dv_try5_inv,${REF:: -${#FILETYPE}}_dv_try5_aligned_inv.tiff] \
        --interpolation WelchWindowedSinc \
        --use-histogram-matching 1 \
        -r [${REF},${TEMPLATE},1] \
        -t SyN[0.2,6,0.1] \
        -m CC[${REF}, ${TEMPLATE}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_REF}, ${MASK_T}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        ;;


    "align_hcr_dv_syn")

        TEMPLATE="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_d.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v_aligned.nrrd"

        echo "$TEMPLATE"
        echo "$REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_lmhcr_try3,${TEMPLATE:: -${#FILETYPE}}_lmhcr_try3_aligned.tiff] \
        --interpolation WelchWindowedSinc \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t SyN[0.05,6,0.2] \
        -m CC[${TEMPLATE}, ${REF}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${REF:: -${#FILETYPE}}_lmhcr_try3_inv,${REF:: -${#FILETYPE}}_lmhcr_try3_aligned_inv.tiff] \
        --interpolation WelchWindowedSinc \
        --use-histogram-matching 1 \
        -r [${REF},${TEMPLATE},1] \
        -t SyN[0.05,6,0.2] \
        -m CC[${REF}, ${TEMPLATE}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_REF}, ${MASK_T}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox
        "
        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        ;;


    "align_juv_hcr_noc2")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        MASK_T=${TEMPLATE/_masked}
        MASK_REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_vmask123.nrrd"
        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$MASK_T"
        echo "$MASK_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_m_,${TEMPLATE:: -${#FILETYPE}}_aligned_m.tiff] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.01,0.99] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 1, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.2,6,0] \
        -m CC[${TEMPLATE}, ${REF}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        -x [${MASK_T}, ${MASK_REF}] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox

        ${ANTSBIN}/antsApplyTransforms \
        -d 3 --verbose 1 \
        -r ${REF} \
        -i ${TEMPLATE} \
        -o ${TEMPLATE%.*}_aligned_r_m.nrrd \
        -t [${TEMPLATE%.*}_m_0GenericAffine.mat , 1] \
        -t ${TEMPLATE%.*}_m_1InverseWarp.nii.gz
        "

        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        done
        ;;


    "align_juv_hcr_nomask")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        C2_T=${TEMPLATE/C1/C2}

        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd.nrrd"
        C2_REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd_cort.nrrd"

        echo "$TEMPLATE"
        echo "$REF"
        echo "$C2_T"
        echo "$C2_REF"

        ANTSCALL="${ANTSBIN}/antsRegistration \
        -d 3 --float 1 --verbose 1 \
        -o [${TEMPLATE:: -${#FILETYPE}}_c2_,${TEMPLATE:: -${#FILETYPE}}_aligned_c2.tiff] \
        --interpolation WelchWindowedSinc \
        --winsorize-image-intensities [0.05,0.95] \
        --use-histogram-matching 1 \
        -r [${TEMPLATE},${REF},1] \
        -t rigid[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 0.7, 32, Regular, 0.25] \
        -m MI[${C2_T}, ${C2_REF}, 0.3, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t Affine[0.1] \
        -m MI[${TEMPLATE}, ${REF}, 0.7, 32, Regular, 0.25] \
        -m MI[${C2_T}, ${C2_REF}, 0.3, 32, Regular, 0.25] \
        -c [200x200x200x0,1e-8,10] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        -t SyN[0.2,6,0] \
        -m CC[${TEMPLATE}, ${REF}, 1, 4] \
        -c [200x200x200x200x10,1e-8,10] \
        --shrink-factors 12x8x4x2x1 \
        --smoothing-sigmas 4x3x2x1x0vox

        ${ANTSBIN}/antsApplyTransforms \
        -d 3 --verbose 1 \
        -r ${REF} \
        -i ${TEMPLATE} \
        -o ${TEMPLATE%.*}_aligned_r_c2.nrrd \
        -t [${TEMPLATE%.*}_c2_0GenericAffine.mat , 1] \
        -t ${TEMPLATE%.*}_c2_1InverseWarp.nii.gz
        "

        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${TEMPLATE:${#OUTPATH}:-${#FILETYPE}}" --wrap="${ANTSCALL}"
        done
        ;;

    "generate_average_brain_mask")

        declare -a MASKS
        for file in ${BRAINSPATH}; do
        echo "$file"
        file=${file/C1/C4}
        file=${file/_masked}
        MASKS=("${MASKS[@]}" "$file")
        echo "${MASKS[@]}";
        done

        ANTSCALL="${ANTSBIN}/antsMultivariateTemplateConstruction2.sh \
        -d 3 \
        -o ${OUTPATH}T_${PREFIX}_ \
        -i 12 \
        -g 0.2 \
        -j 32 \
        -v 500 \
        -c 2 \
        -k 1 \
        -w 1 \
        -f 4x2 \
        -s 2x1 \
        -q 100x100 \
        -n 0 \
        -r 1 \
        -l 1 \
        -m CC[2] \
        -t SyN[0.3,6,0.0]\
        -x ${MASKS[@]}\
        ${BRAINSPATH}
        "
        echo "$BRAINSPATH"
        echo "${MASKS[@]}"

        sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-template --wrap="${ANTSCALL}"
        ;;

    "apply-reverse-hcr")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd.nrrd"
        CHANNELS=${TEMPLATE/C1-/*}

        echo "$TEMPLATE"
        echo "$CHANNELS"
        for CHANNEL in ${CHANNELS}; do \
        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "${REF}" \
        -i "${CHANNEL}" \
        -o "${CHANNEL%.*}_aligned_ref_mc2.nrrd" \
        -t ["${TEMPLATE%.*}_mc2_0GenericAffine.mat", 1]\
        -t "${TEMPLATE%.*}_mc2_1InverseWarp.nii.gz";
        done;
        # sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-applyrev-"${CHANNEL}" --wrap="${ANTSCALL}"
        done
        ;;

      "apply-aligned-hcr-2p")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        REF="/raven/u/${USER}/reg/fixed/21dpf_AVG_H2BGCaMP6s.nrrd"
        CHANNELS=${TEMPLATE/C1-/*}

        echo "$TEMPLATE"
        echo "$CHANNELS"
        for CHANNEL in ${CHANNELS}; do \
        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "${REF}" \
        -i "${CHANNEL}" \
        -o "${CHANNEL%.*}_2p.nrrd" \
        -t "/raven/u/${USER}/reg/fixed/21dpf_AVG_H2BGCaMP6s_try31Warp.nii.gz" \
        -t "/raven/u/${USER}/reg/fixed/21dpf_AVG_H2BGCaMP6s_try30GenericAffine.mat";
        done;
        done
        ;;

      "apply-aligned-hcr-2p-rev")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        REF="/raven/u/${USER}/reg/fixed/21dpf_AVG_H2BGCaMP6s.nrrd"
        CHANNELS=${TEMPLATE/C1-/*}

        echo "$TEMPLATE"
        echo "$CHANNELS"
        for CHANNEL in ${CHANNELS}; do \
        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "${REF}" \
        -i "${CHANNEL}" \
        -o "${CHANNEL%.*}_gamma2p.nrrd" \
        -t "[/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd_gamma04_zres2_gammazres2_resc2_980GenericAffine.mat, 1]"\
        -t "/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_vd_gamma04_zres2_gammazres2_resc2_981InverseWarp.nii.gz" ;
        done;
        done
        ;;

    "apply-reverse-hcr-v")
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        REF="/raven/u/${USER}/reg/fixed/juvenile_hcr_ref_dv_v.nrrd"
        CHANNELS=${TEMPLATE/C1-/*}

        echo "$TEMPLATE"
        echo "$CHANNELS"
        for CHANNEL in ${CHANNELS}; do \
        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "${REF}" \
        -i "${CHANNEL}" \
        -o "${CHANNEL%.*}_aligned_ref.nrrd" \
        -t ["${TEMPLATE%.*}_mc2_0GenericAffine.mat", 1]\
        -t "${TEMPLATE%.*}_mc2_1InverseWarp.nii.gz";
        done;
        # sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-applyrev-"${CHANNEL}" --wrap="${ANTSCALL}"
        done
        ;;

    "apply-reverse-7dpf") # TODO: write sbatch command
        REF="/raven/u/${USER}/reg/fixed/7dpf_AVG_H2BGCaMP6s_atlas.nrrd"

        for TEMPLATE in /raven/u/${USER}/reg/moving_one/${PREFIX}; do

        echo "$TEMPLATE"

        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "${REF}" \
        -i "${TEMPLATE}" \
        -o "${TEMPLATE%.*}_aligned_ref_mc2.nrrd" \
        -t ["${TEMPLATE%.*}0GenericAffine.mat", 1]\
        -t "${TEMPLATE%.*}1InverseWarp.nii.gz"
        done
        ;;

    "apply-vtod") # TODO: write sbatch command
        echo "hello"
        PREFIX="*m_masked.tif_aligned_ref.nrrd"
        echo "$PREFIX"
        for TEMPLATE in /raven/u/${USER}/reg/moving_hcr/${PREFIX}; do

        echo "$TEMPLATE"

        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "/raven/u/jkap/reg/fixed/juvenile_hcr_ref_dv_merge_clip_gamma.nrrd" \
        -i "${TEMPLATE}" \
        -o "${TEMPLATE%.*}_todv.nrrd" \
        -t "/raven/u/jkap/reg/fixed/df_hcr.nii.gz" \
	      -t "/raven/u/jkap/reg/fixed/affine_hcr.txt"
        done
        ;;

    "apply-reverse-21dpf") # TODO: write sbatch command
        REF="/raven/u/${USER}/reg/fixed/21dpf_AVG_H2BGCaMP6s.nrrd"

        for TEMPLATE in /raven/u/${USER}/reg/moving_three/${PREFIX}; do

        echo "$TEMPLATE"

        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "${REF}" \
        -i "${TEMPLATE}" \
        -o "${TEMPLATE%.*}_aligned_ref_mc2.nrrd" \
        -t ["${TEMPLATE%.*}0GenericAffine.mat", 1]\
        -t "${TEMPLATE%.*}1InverseWarp.nii.gz"
        done
        ;;

    "apply-forward") # TODO: write sbatch command
        for TEMPLATE in $(find "/raven/${USER}/reg/moving/" -name "*.nrrd"); do
        "${ANTSBIN}/antsApplyTransforms" -d 3 --verbose 1 \
        -r "${template}" \
        -i "${REF}" \
        -o "${template%.*}_aligned.nrrd" \
        -t "${template%.*}_1Warp.nii.gz"\
        -t "${template%.*}_0GenericAffine.mat"
        done
        ;;

    "transform-points")  # assumes format YYYYMMDD_fish*_SUFFIX*.csv
      for POINTS in ${POINTPATH}/*; do
      USC1=$(expr index "$POINTS" '_')
      USC2=$(expr index "${POINTS:USC1}" '_')
      ANIMAL=${POINTS::USC1+USC2}
      WARP="${OUTPATH_POINTS::-1}${ANIMAL:${#POINTPATH}}${SUFFIX_POINTS%.*}1Warp.nii.gz"
      AFFINE="${OUTPATH_POINTS::-1}${ANIMAL:${#POINTPATH}}${SUFFIX_POINTS%.*}0GenericAffine.mat"
      ANTSCALL="${ANTSBIN}/antsApplyTransformsToPoints \
      -d 3 \
      -i ${POINTS} \
      -o ${POINTS%.*}_aligned.csv \
      -t ${WARP} \
      -t ${AFFINE}"

      if test -f "$WARP"; then
        echo "$WARP exists."
      fi
      if test -f "$AFFINE"; then
        echo "$AFFINE exists."
      fi
      if test -f "$POINTS"; then
        echo "$POINTS exists."
      fi

      sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${POINTS:${#POINTPATH}:-1}" --wrap="${ANTSCALL}"
      done
      ;;

    "transform-points-reverse")  # assumes format YYYYMMDD_fish*_SUFFIX*.csv
      for POINTS in ${POINTPATH}/*; do
      USC1=$(expr index "$POINTS" '_')
      USC2=$(expr index "${POINTS:USC1}" '_')
      ANIMAL=${POINTS::USC1+USC2}
      INVWARP="${OUTPATH_POINTS::-1}${ANIMAL:${#POINTPATH}}${SUFFIX_POINTS%.*}1InverseWarp.nii.gz"
      AFFINE="${OUTPATH_POINTS::-1}${ANIMAL:${#POINTPATH}}${SUFFIX_POINTS%.*}0GenericAffine.mat"
      ANTSCALL="${ANTSBIN}/antsApplyTransformsToPoints \
      -d 3 \
      -i ${POINTS} \
      -o ${POINTS%.*}_aligned.csv \
      -t [${AFFINE}, 1] \
      -t ${INVWARP}"

      if test -f "$INVWARP"; then
        echo "$INVWARP exists."
      else
        echo "$INVWARP does not exists."
      fi
      if test -f "$AFFINE"; then
        echo "$AFFINE exists."
      else
        echo "$AFFINE does not exists."
      fi

      sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${POINTS:${#POINTPATH}:-1}" --wrap="${ANTSCALL}"
      done
      ;;

    "transform-points-21dpf")  # assumes format YYYYMMDD_fish*_SUFFIX*.csv
      for POINTS in ${POINTPATH_THREE}/${PREFIX}; do

      USC1=$(expr index "$POINTS" '_')
      USC2=$(expr index "${POINTS:USC1}" '_')
      USC3=$(expr index "${POINTS:USC1+USC2}" '_')

      ANIMAL=${POINTS::USC1+USC2+USC3}
      WARP="${OUTPATH_THREE}${ANIMAL:${#POINTPATH_THREE}}${SUFFIX_POINTS%.*}1Warp.nii.gz"
      AFFINE="${OUTPATH_THREE}${ANIMAL:${#POINTPATH_THREE}}${SUFFIX_POINTS%.*}0GenericAffine.mat"

      if test -f "$WARP"; then
        echo "$WARP exists."
      else
        echo "$WARP does not exists."
      fi
      if test -f "$AFFINE"; then
        echo "$AFFINE exists."
      else
        echo "$AFFINE does not exists."
      fi

      ANTSCALL= $"${ANTSBIN}/antsApplyTransformsToPoints" \
      -d 3 \
      -i ${POINTS} \
      -o ${POINTS%.*}_aligned_ref.csv \
      -t ${WARP} \
      -t ${AFFINE}
      # sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${POINTS:${#POINTPATH}:-1}" --wrap="${ANTSCALL}"
      done
      ;;

    "transform-points-7dpf")  # assumes format YYYYMMDD_fish*_SUFFIX*.csv
      for POINTS in ${POINTPATH_ONE}/${PREFIX}; do

      USC1=$(expr index "$POINTS" '_')
      USC2=$(expr index "${POINTS:USC1}" '_')
      USC3=$(expr index "${POINTS:USC1+USC2}" '_')

      ANIMAL=${POINTS::USC1+USC2+USC3}
      WARP="${OUTPATH_ONE}${ANIMAL:${#POINTPATH_ONE}}${SUFFIX_POINTS%.*}1Warp.nii.gz"
      AFFINE="${OUTPATH_ONE}${ANIMAL:${#POINTPATH_ONE}}${SUFFIX_POINTS%.*}0GenericAffine.mat"

      if test -f "$WARP"; then
        echo "$WARP exists."
      else
        echo "$WARP does not exists."
      fi
      if test -f "$AFFINE"; then
        echo "$AFFINE exists."
      else
        echo "$AFFINE does not exists."
      fi

      ANTSCALL= $"${ANTSBIN}/antsApplyTransformsToPoints" \
      -d 3 \
      -i ${POINTS} \
      -o ${POINTS%.*}_aligned.csv \
      -t ${WARP} \
      -t ${AFFINE}
      # sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${POINTS:${#POINTPATH}:-1}" --wrap="${ANTSCALL}"
      done
      ;;

    "transform-points-7dpf-hcrtoref")  # assumes format YYYYMMDD_fish*_SUFFIX*.csv
      for POINTS in ${POINTPATH_ONE}/${PREFIX}; do

      USC1=$(expr index "$POINTS" '_')
      USC2=$(expr index "${POINTS:USC1}" '_')
      USC3=$(expr index "${POINTS:USC1+USC2}" '_')

      ANIMAL=${POINTS::USC1+USC2+USC3}
      INVWARP="${OUTPATH_ONE}${ANIMAL:${#POINTPATH_ONE}}${SUFFIX_HCRTOREF%.*}_1Warp.nii.gz"
      AFFINE="${OUTPATH_ONE}${ANIMAL:${#POINTPATH_ONE}}${SUFFIX_HCRTOREF%.*}_0GenericAffine.mat"
      echo "$ANIMAL"
      if test -f "$INVWARP"; then
        echo "$INVWARP exists."
      else
        echo "$INVWARP does not exists."
      fi
      if test -f "$AFFINE"; then
        echo "$AFFINE exists."
      else
        echo "$AFFINE does not exists."
      fi

      ANTSCALL= $"${ANTSBIN}/antsApplyTransformsToPoints" \
      -d 3 \
      -i ${POINTS} \
      -o ${POINTS%.*}_hcrref.csv \
      -t [${AFFINE}, 1] \
      -t ${INVWARP}
      # sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${POINTS:${#POINTPATH}:-1}" --wrap="${ANTSCALL}"
      done
      ;;

    "transform-points-reverse-7dpf")  # assumes format YYYYMMDD_fish*_SUFFIX*.csv
      for POINTS in ${POINTPATH_ONE}/${PREFIX}; do

      echo "$POINTS"

      USC1=$(expr index "$POINTS" '_')
      USC2=$(expr index "${POINTS:USC1}" '_')
      USC3=$(expr index "${POINTS:USC1+USC2}" '_')

      ANIMAL=${POINTS::USC1+USC2+USC3}
      echo "$ANIMAL"
      INVWARP="${OUTPATH_ONE}${ANIMAL:${#POINTPATH_ONE}}${SUFFIX_POINTS%.*}1InverseWarp.nii.gz"
      AFFINE="${OUTPATH_ONE}${ANIMAL:${#POINTPATH_ONE}}${SUFFIX_POINTS%.*}0GenericAffine.mat"

      if test -f "$INVWARP"; then
        echo "$INVWARP exists."
      else
        echo "$INVWARP does not exist."
      fi
      if test -f "$AFFINE"; then
        echo "$AFFINE exists."
      else
        echo "$AFFINE does not exist."
      fi

      ANTSCALL= $"${ANTSBIN}/antsApplyTransformsToPoints" \
      -d 3 \
      -i ${POINTS} \
      -o ${POINTS%.*}_aligned.csv \
      -t [${AFFINE}, 1] \
      -t ${INVWARP}

      # sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${POINTS:${#POINTPATH}:-1}" --wrap="${ANTSCALL}"
      done
      ;;

    "transform-points-reverse-21dpf")  # assumes format YYYYMMDD_fish*_SUFFIX*.csv
      for POINTS in ${POINTPATH_THREE}/${PREFIX}; do

      USC1=$(expr index "$POINTS" '_')
      USC2=$(expr index "${POINTS:USC1}" '_')
      USC3=$(expr index "${POINTS:USC1+USC2}" '_')

      ANIMAL=${POINTS::USC1+USC2+USC3}
      INVWARP="${OUTPATH_THREE}${ANIMAL:${#POINTPATH_THREE}}${SUFFIX_POINTS%.*}1InverseWarp.nii.gz"
      AFFINE="${OUTPATH_THREE}${ANIMAL:${#POINTPATH_THREE}}${SUFFIX_POINTS%.*}0GenericAffine.mat"

      if test -f "$INVWARP"; then
        echo "$INVWARP exists."
      else
        echo "$INVWARP does not exists."
      fi
      if test -f "$AFFINE"; then
        echo "$AFFINE exists."
      else
        echo "$AFFINE does not exists."
      fi

      ANTSCALL= $"${ANTSBIN}/antsApplyTransformsToPoints" \
      -d 3 \
      -i ${POINTS} \
      -o ${POINTS%.*}_aligned.csv \
      -t [${AFFINE}, 1] \
      -t ${INVWARP}
      # sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-reg-"${POINTS:${#POINTPATH}:-1}" --wrap="${ANTSCALL}"
      done
      ;;

    "generate_average_brain")
      ANTSCALL="${ANTSBIN}/antsMultivariateTemplateConstruction2.sh \
      -d 3 \
      -o ${OUTPATH}T_${PREFIX}_ \
      -i 12 \
      -g 0.2 \
      -j 32 \
      -v 500 \
      -c 2 \
      -k 1 \
      -w 1 \
      -f 4x2 \
      -s 2x1 \
      -q 100x100 \
      -n 0 \
      -r 1 \
      -l 1 \
      -m CC[2] \
      -t SyN[0.1,6,0.0]\
      ${BRAINSPATH}
      "
      echo "HELLO"
      echo "${BRAINSPATH}"

      #sbatch --partition="$PARTITION" -N 1 -n 1 -t "$WALL_TIME" -J ANTs-template --wrap="${ANTSCALL}" --mem=120000

      sbatch --partition="$PARTITION" -N 1 -n 1 -c 72 -t "$WALL_TIME" -J ANTs-template --wrap="${ANTSCALL}" --mem=256000

    esac


