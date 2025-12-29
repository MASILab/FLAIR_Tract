#!/usr/bin/env bash

T1_FILE=$1
T1_DIR=$(dirname "$T1_FILE")
RESOURCE_DIR='/fs5/p_masi/schwat1/spie_flair_tract_extension/resources/'
PREPROC_DIR='/fs5/p_masi/schwat1/spie_flair_tract_extension/preprocessing/'
MRtrix="$RESOURCE_DIR/sifs/MRtrix3.sif"
ANTS="$RESOURCE_DIR/sifs/Ants.sif"
UNEST="$RESOURCE_DIR/sifs/Unest.sif"
TRACTSEG="/fs5/p_masi/schwat1/flair_tractography/sifs/tractSeg_release_cpu.simg"
OUT_5TTGEN="$T1_DIR/T1_5tt.nii.gz"
OUT_FSLMATHS="$T1_DIR/T1_seed.nii.gz"
OUT_GMWMI="$T1_DIR/T1_gmwmi.nii.gz"
OUT_FSLMERGE="$T1_DIR/T1_tractseg.nii.gz"
WML_DIR="$T1_DIR/wm_learning"
THREADS=48

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS

apptainer run -e --contain --home "$T1_DIR" --bind "$T1_DIR":/INPUTS --bind "$T1_DIR":/OUTPUTS --bind \
"$RESOURCE_DIR/working_dir":/WORKING_DIR --bind /tmp:/tmp \
--bind "$RESOURCE_DIR" "$UNEST" --w_skull 

#cp "$T1_DIR/FinalResults/wholebrain/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm_seg.nii.gz" "$T1_DIR/T1_2pt2mm_seg.nii.gz" || exit
cp "$T1_DIR/FinalResults/wholebrain/mni_icbm152_t1_tal_nlin_sym_09c_seg.nii.gz" "$T1_DIR/T1_seg.nii.gz" || exit

fslmaths "$T1_DIR/T1_seg.nii.gz" -div "$T1_DIR/T1_seg.nii.gz" -fillh "$T1_DIR/T1_mask.nii.gz" -odt int || exit

apptainer exec -ce --bind "$T1_DIR" "$ANTS" N4BiasFieldCorrection -d 3 -i "$T1_FILE" -x "$T1_DIR/T1_mask.nii.gz" -o "$T1_DIR/T1_N4.nii.gz" || exit 

apptainer exec -e --contain --bind /tmp:/tmp --bind "$RESOURCE_DIR" --bind "$PREPROC_DIR" --home /tmp "$MRtrix" 5ttgen fsl -nthreads "$THREADS" "$T1_DIR/T1_N4.nii.gz" "$OUT_5TTGEN" -mask "$T1_DIR/T1_mask.nii.gz" -nocrop -v || exit

# # Generate seed maps:
# # - T1_seed.nii.gz
# # - T1_gmwmi.nii.gz

echo "prep_T1.sh: Computing seed mask..."
fslmaths "$OUT_5TTGEN" -roi 0 -1 0 -1 0 -1 2 1 -bin -Tmax "$OUT_FSLMATHS" -odt int || exit

apptainer exec -e --contain --bind "$RESOURCE_DIR" --bind "$PREPROC_DIR" "$MRtrix" 5tt2gmwmi -mask_in "$T1_DIR/T1_mask.nii.gz" "$OUT_5TTGEN" "$OUT_GMWMI" || exit

# # Generate bundle priors:
# # - T1_tractseg.nii.gz

# echo "prep_T1.sh: Computing bundle priors..."
mkdir -p "$WML_DIR"
apptainer run --contain -e --bind /tmp:/tmp --bind "$T1_DIR/T1_N4.nii.gz":/INPUTS/T1.nii.gz --bind "$WML_DIR":/OUTPUTS \
--env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$TRACTSEG"

fslmerge -t "$OUT_FSLMERGE" "$WML_DIR"/tractSeg/orig/*.nii.gz # without globbing, fslmerge errors w/ filename too long

apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsRegistrationSyN.sh -d 3 -m "$T1_DIR/T1_N4.nii.gz" -f "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -t r -o "$T1_DIR/mni_t1_" || exit
mv "$T1_DIR/mni_t1_Warped.nii.gz" "$T1_DIR/T1_N4_mni_1mm.nii.gz"
rm "$T1_DIR/mni_t1_InverseWarped.nii.gz"
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsRegistrationSyN.sh -d 3 -m "$T1_DIR/T1_N4.nii.gz" -f "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -t r -o "$T1_DIR/mni_t1_2pt2mm_" || exit
#mv "$T1_DIR/mni_t1_2pt2mm_Warped.nii.gz" "$T1_DIR/T1_N4_mni_2pt2mm.nii.gz"
#rm "$T1_DIR/mni_t1_2pt2mm_InverseWarped.nii.gz"
# ==========================================
# TODO
# MAKE SURE T12MNI IS GENERATED FOR 1mm
#
# T12MNI -> mni_t1_
#
# ==========================================

echo "prep_T1.sh: Moving images to MNI space at 1mm isotropic..."
apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -i "$T1_DIR/T1_mask.nii.gz" -t $T1_DIR/mni_t1_0GenericAffine.mat -o $T1_DIR/T1_mask_mni_1mm.nii.gz -n NearestNeighbor
apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -i "$T1_DIR/T1_seed.nii.gz" -t $T1_DIR/mni_t1_0GenericAffine.mat -o $T1_DIR/T1_seed_mni_1mm.nii.gz -n NearestNeighbor
apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -i "$T1_DIR/T1_5tt.nii.gz"  -t $T1_DIR/mni_t1_0GenericAffine.mat -o $T1_DIR/T1_5tt_mni_1mm.nii.gz  -n Linear
apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -i "$T1_DIR/T1_gmwmi.nii.gz" -t $T1_DIR/mni_t1_0GenericAffine.mat -o $T1_DIR/T1_gmwmi_mni_1mm.nii.gz -n Linear
apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -i "$T1_DIR/T1_seg.nii.gz"  -t $T1_DIR/mni_t1_0GenericAffine.mat -o $T1_DIR/T1_seg_mni_1mm.nii.gz  -n NearestNeighbor
apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -i "$T1_DIR/T1_tractseg.nii.gz"  -t $T1_DIR/mni_t1_0GenericAffine.mat -o $T1_DIR/T1_tractseg_mni_1mm.nii.gz -n Linear
# apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz" -i "$T1_DIR/T1_slant.nii.gz"  -t $T1_DIR/mni_t1_0GenericAffine.mat -o $T1_DIR/T1_slant_mni_1mm.nii.gz -n NearestNeighbor

#echo "prep_T1.sh: Moving images to MNI space at 2pt2mm isotropic..."
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -i "$T1_DIR/T1_N4.nii.gz" -t "$T1_DIR/mni_t1_2pt2mm_0GenericAffine.mat" -o "$T1_DIR/T1_N4_mni_2pt2mm.nii.gz" -n Linear
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -i "$T1_DIR/T1_mask.nii.gz" -t "$T1_DIR/mni_t1_2pt2mm_0GenericAffine.mat" -o "$T1_DIR/T1_mask_mni_2pt2mm.nii.gz" -n NearestNeighbor
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -i "$T1_DIR/T1_seed.nii.gz" -t "$T1_DIR/mni_t1_2pt2mm_0GenericAffine.mat" -o "$T1_DIR/T1_seed_mni_2pt2mm.nii.gz" -n NearestNeighbor -v || exit
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 3 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -i "$T1_DIR/T1_5tt.nii.gz" -t "$T1_DIR/mni_t1_2pt2mm_0GenericAffine.mat" -o "$T1_DIR/T1_5tt_mni_2pt2mm.nii.gz" -n Linear
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -i "$T1_DIR/T1_gmwmi.nii.gz" -t "$T1_DIR/mni_t1_2pt2mm_0GenericAffine.mat" -o "$T1_DIR/T1_gmwmi_mni_2pt2mm.nii.gz" -n Linear
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 0 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -i "$T1_DIR/T1_seg.nii.gz" -t "$T1_DIR/mni_t1_2pt2mm_0GenericAffine.mat" -o "$T1_DIR/T1_seg_mni_2pt2mm.nii.gz" -n NearestNeighbor
#apptainer exec -e --contain --bind "$RESOURCE_DIR" --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS" "$ANTS" antsApplyTransforms -d 3 -e 3 -r "$T1_DIR/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz" -i "$T1_DIR/T1_tractseg.nii.gz" -t "$T1_DIR/mni_t1_2pt2mm_0GenericAffine.mat" -o "$T1_DIR/T1_tractseg_mni_2pt2mm.nii.gz" -n Linear
# Should FinalResults/<nested dir>/segmentation be used here?
# antsApplyTransforms -d 3 -e 3 -r $T1_DIR/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_2pt2mm.nii.gz -i $T1_DIR/T1_slant.nii.gz -t $T1_DIR/T12mni_0GenericAffine.mat -o $T1_DIR/T1_slant_mni_2pt2mm.nii.gz -n NearestNeighbor

printf "%s\n" "T1 Processing Finished"
