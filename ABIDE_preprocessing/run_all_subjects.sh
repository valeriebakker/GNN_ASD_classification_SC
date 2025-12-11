#!/usr/bin/env bash
set -euo pipefail

# Path to the preprocessing script
PIPELINE=/home/vbakker/code/preproc_and_tractography.sh

# Root dataset folder (where all ABIDEII-* sites live)
ROOT=/home/vbakker/data_4sites

# Loop through all sites you want
for site in ABIDEII-NYU_1 ABIDEII-NYU_2 ABIDEII-SDSU_1 ABIDEII-TCD_1; do
    BASEDIR="$ROOT/$site"
    [[ -d "$BASEDIR" ]] || continue

    for subj in "$BASEDIR"/*; do
        dti_dir="$subj/session_1/dti_1"
        [[ -d "$dti_dir" ]] || continue

        t1_dir="$subj/session_1/anat_1"
        t1_img="$t1_dir/anat.nii.gz"

        subj_id=$(basename "$subj")
        echo "RUNNING $site / $subj_id"

        # Create site-specific acqparams.txt before running pipeline
        case "$site" in
            ABIDEII-NYU_1)  # PE-direction = RL
                echo "1 0 0 0.02016" > "$dti_dir/acqparams.txt"
                ;;
            ABIDEII-NYU_2)  # PE-direction = RL
                echo "1 0 0 0.02016" > "$dti_dir/acqparams.txt"
                ;;
            ABIDEII-SDSU_1) # PE-direction = RL
                echo "1 0 0 0.050" > "$dti_dir/acqparams.txt"
                ;;
            ABIDEII-TCD_1)  # PE-direction = AP
                echo "0 -1 0 0.050" > "$dti_dir/acqparams.txt"
                ;;
            *)
                echo "Unknown site: $site"
                exit 1
                ;;
        esac

        # Run pipeline (check for T1-image)
        if [[ -f "$t1_img" ]]; then
            echo "Found T1 image: $t1_img"
            bash "$PIPELINE" "$dti_dir" "$t1_img"
        else
            echo "No T1 image found for $site / $subj_id. Running without T1."
            bash "$PIPELINE" "$dti_dir"
        fi

        # If error occurs
        if [[ $? -ne 0 ]]; then
            echo "!!! ERROR: pipeline failed for $site / $subj_id"
            exit 1
        fi
    done
done
