from args import Args
import utils as Reader
from sklearn.model_selection import StratifiedShuffleSplit

from neuroHarmonize import harmonizationLearn, harmonizationApply

import numpy as np
import pandas as pd
import os

### COMBAT HARMONIZATION (only when args.use_combat = True) ###

def harmonize_connectomes_with_combat(
        dev_features, test_features,
        dev_subject_IDs, test_subject_IDs,
        pheno_csv_path):

    def build_batch(subject_ids):
        subject_ids = np.asarray(subject_ids).astype(str)

        # Get the site ID for each subject
        site_dict = Reader.collect_pheno_scores_in_dict(
            pheno_csv_path, subject_ids, 'SITE_ID')

        # Build a list of site labels in the same order as subject_ids
        site_list = [str(site_dict[sid]) for sid in subject_ids]

        # Convert site names to integer codes
        unique_sites = sorted(set(site_list))
        site_to_int = {s: i for i, s in enumerate(unique_sites)}

        # Numeric batch vector (one integer per subject)
        batch_numeric = np.array([site_to_int[s] for s in site_list], dtype=int)

        # Create a DataFrame with the batch variable
        covars = pd.DataFrame({"SITE": batch_numeric}, index=subject_ids)

        return covars

    # Build batch covariate tables
    dev_covars = build_batch(dev_subject_IDs)
    test_covars = build_batch(test_subject_IDs)

    print("ComBat batches in dev set:", np.unique(dev_covars["SITE"]))
    print("ComBat batches in test set:", np.unique(test_covars["SITE"]))

    # Fit ComBat on development set only
    combat_model, dev_h = harmonizationLearn(
        dev_features,
        covars=dev_covars)

    # Apply trained ComBat model to test set
    test_h = harmonizationApply(
        test_features,
        covars=test_covars,
        model=combat_model)

    return dev_h, test_h


### LOAD CONNECTIVITY DATA IN (+ OPTIONALLY PHENO DATA) ###

def load_data_and_process():
    args = Args()

    # Input data variables
    data_folder = os.path.join('C:/Users/20202932/8STAGE/data_4sites')
    pheno_csv_path = os.path.join(data_folder, 'pheno_dataset.csv')  # Should at least have DX_GROUP, SEX, SITE_ID, AGE_AT_SCAN
    acq_sites = ["ABIDEII-NYU_1", "ABIDEII-NYU_2", "ABIDEII-SDSU_1", "ABIDEII-TCD_1"]

    # Option 1: Leave-One-Site-Out split
    if args.split_mode.lower() == "loso":

        print("\nUsing Leave-One-Site-Out (LOSO) split")

        dev_sites = ["ABIDEII-NYU_1", "ABIDEII-NYU_2", "ABIDEII-SDSU_1"]
        test_sites = ["ABIDEII-TCD_1"]      # Or choose a different site

        # Load all subjects for development set and test set
        dev_subjects, dev_paths = Reader.collect_subjects_of_sites(data_folder, dev_sites)
        test_subjects, test_paths = Reader.collect_subjects_of_sites(data_folder, test_sites)

        dev_subject_IDs = np.array(dev_subjects)
        test_subject_IDs = np.array(test_subjects)

        # Load structural features (flattened upper triangle)
        dev_features = Reader.load_matrices_and_flatten(dev_paths)
        test_features = Reader.load_matrices_and_flatten(test_paths)

        # Load labels (DX_GROUP, coded 1/2 in pheno, so converted to 0/1)
        dev_y_true_dict = Reader.collect_pheno_scores_in_dict(pheno_csv_path, dev_subject_IDs, 'DX_GROUP')
        test_y_true_dict = Reader.collect_pheno_scores_in_dict(pheno_csv_path, test_subject_IDs, 'DX_GROUP')

        dev_y_true = (np.array([int(dev_y_true_dict[s]) for s in dev_subject_IDs]) - 1).astype(int)
        test_y_true = (np.array([int(test_y_true_dict[s]) for s in test_subject_IDs]) - 1).astype(int)

    # Option 2: Mixed stratified split
    elif args.split_mode.lower() == "mixed":

        print("\nUsing mixed stratified split across all sites")

        # Load all subjects from all sites
        all_subjects, all_paths = Reader.collect_subjects_of_sites(data_folder, acq_sites)
        all_subject_IDs = np.array(all_subjects)

        # Load all structural features (flattened upper triangle)
        all_features = Reader.load_matrices_and_flatten(all_paths)

        # Load all labels (DX_GROUP, coded 1/2 in pheno, so converted to 0/1)
        all_y_true_dict = Reader.collect_pheno_scores_in_dict(pheno_csv_path, all_subject_IDs, 'DX_GROUP')
        all_y_true = (np.array([int(all_y_true_dict[s]) for s in all_subject_IDs]) - 1).astype(int)


        # ###################
        # # Below is a sanity check to train on evenly balanced classes.
        # # To use the full dataset, keep this section commented out.

        # # BALANCE ASD AND TDC BEFORE SPLITTING

        # # Get indices for ASD and TDC subjects
        # asd_idx = np.where(all_y_true == 0)[0]   # ASD = 0
        # tdc_idx = np.where(all_y_true == 1)[0]  # TDC = 1

        # n_tdc = len(tdc_idx)    # Number of TDC (68)
        # target_n = n_tdc         # We want the same amount for ASD

        # print(f"Balancing dataset to {target_n} ASD and {n_tdc} TDC.")

        # # For downsampling ASD, check how ASD subjects are distributed across sites
        # asd_sites = np.array([
        #     Reader.collect_pheno_scores_in_dict(pheno_csv_path, [sid], 'SITE_ID')[sid]
        #     for sid in all_subject_IDs[asd_idx]])

        # unique_sites, counts_per_site = np.unique(asd_sites, return_counts=True)
        # print("ASD distribution across sites:", dict(zip(unique_sites, counts_per_site)))

        # # Compute how many ASD subjects to sample per site
        # fractions = (counts_per_site / len(asd_idx)) * target_n

        # # Convert to integer counts
        # int_counts = np.floor(fractions).astype(int)

        # # Number of ASD samples still needed after rounding
        # leftover = target_n - int_counts.sum()

        # # Assign leftover samples to the sites with largest fractional parts
        # fractional_parts = fractions - np.floor(fractions)
        # leftover_sites = np.argsort(fractional_parts)[-leftover:]
        # int_counts[leftover_sites] += 1

        # print("Final ASD allocation per site:", dict(zip(unique_sites, int_counts)))

        # # Randomly select ASD subjects from each site according to the integer counts
        # asd_keep_indices = []
        # for site, n_sample in zip(unique_sites, int_counts):
        #     site_asd_indices = asd_idx[asd_sites == site]
        #     chosen = np.random.choice(site_asd_indices, size=n_sample, replace=False)
        #     asd_keep_indices.extend(chosen)

        # asd_keep_indices = np.array(asd_keep_indices)
        # print(f"Selected ASD: {len(asd_keep_indices)}")

        # # Keep all control subjects
        # tdc_keep_indices = tdc_idx

        # # Combine ASD and control indices to form the balanced subset
        # balanced_idx = np.concatenate([tdc_keep_indices, asd_keep_indices])
        # balanced_idx.sort()

        # # Apply the balanced subset to the full dataset
        # all_subject_IDs = all_subject_IDs[balanced_idx]
        # all_features = all_features[balanced_idx]
        # all_y_true = all_y_true[balanced_idx]

        # print("Balanced total dataset size:", len(all_y_true))
        # print("Final counts -> ASD:", sum(all_y_true == 0),
        #       "Control:", sum(all_y_true == 1))

        # ##############
        
        # Load sites for all subjects
        site_dict = Reader.collect_pheno_scores_in_dict(pheno_csv_path, all_subject_IDs, 'SITE_ID')
        all_sites = np.array([site_dict[s] for s in all_subject_IDs])

        # Create labels that combine diagnosis and site
        stratify_labels = np.array([f"{label}_{site}" for label, site in zip(all_y_true, all_sites)])

        # Stratified split on diagnosis and site
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_percentage, random_state=42)
        dev_idx, test_idx = next(sss.split(all_features, stratify_labels))

        dev_subject_IDs = all_subject_IDs[dev_idx]
        test_subject_IDs = all_subject_IDs[test_idx]

        print(dev_subject_IDs)
        print(test_subject_IDs)

        dev_features = all_features[dev_idx]
        test_features = all_features[test_idx]

        dev_y_true = all_y_true[dev_idx]
        test_y_true = all_y_true[test_idx]

    else:
        raise ValueError("args.split_mode must be 'loso' or 'mixed'")
    
    print(f"Development set: Loaded SC features: {dev_features.shape[0]} subjects x {dev_features.shape[1]} features")
    print(f"Labels distribution of development set: {np.bincount(dev_y_true)}")
    print(f"Test set: Loaded SC features: {test_features.shape[0]} subjects x {test_features.shape[1]} features")
    print(f"Labels distribution of test set: {np.bincount(test_y_true)}")

    # Identify zero-variance or near-zero-variance features in development set
    var_dev = dev_features.var(axis=0)

    # Mask to keep only features with variance above a tiny threshold
    mask = var_dev > 1e-6

    # Remove zero-variance features
    dev_features = dev_features[:, mask]
    test_features = test_features[:, mask]

    # Optional ComBat harmonization of connectivity features
    if getattr(args, "use_combat", False):
        print("\nRunning ComBat harmonization on connectivity matrices")
        dev_features, test_features = harmonize_connectomes_with_combat(
            dev_features, test_features,
            dev_subject_IDs, test_subject_IDs,
            pheno_csv_path)

    # Check for use of phenotypic data for graph construction
    if args.use_pheno_data:
        # Compute affinity matrix of phenotypic similarity based on sex and age
        pheno_affinity_matrix = Reader.create_affinity_matrix_from_pheno_scores(dev_subject_IDs, pheno_csv_path)
    else:
        pheno_affinity_matrix = None

    return (
        args,
        dev_subject_IDs, test_subject_IDs,
        dev_features, test_features,
        dev_y_true, test_y_true,
        pheno_affinity_matrix)