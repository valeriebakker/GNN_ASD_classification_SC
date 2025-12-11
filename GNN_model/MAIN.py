import os
import pandas as pd
import torch
import numpy as np
import random
from data_processing import load_data_and_process
from train_and_test import train_gnn_cv, refit_final_model, test_gnn_inductive
from utils import cross_affinity_matrix_from_pheno_scores

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Select compute device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of folds used for cross-validation
n_folds = 5

# Load data and construct graphs
args, dev_subject_IDs, test_subject_IDs, dev_features, test_features, \
    dev_y_true, test_y_true, pheno_affinity_matrix = load_data_and_process()

# Cross-validation to select hyperparameters
model, fold_results, mean_results, results_dir = train_gnn_cv(
    args,
    dev_features,
    dev_y_true,
    pheno_affinity_matrix,
    n_folds,
    device)

# Retrain final model on all development data
final_model, node_ftr, selected_idx, final_model_results = refit_final_model(
    args,
    dev_features,
    dev_y_true,
    pheno_affinity_matrix,
    results_dir,
    device)

# Check if phenotypic data is used for graph edges
if args.use_pheno_data:

    # Compute cross phenotypic similarity between test and development set
    data_folder = os.path.join('C:/Users/20202932/8STAGE/data_4sites')
    pheno_csv_path = os.path.join(data_folder, 'pheno_dataset.csv')

    pheno_test_to_dev = cross_affinity_matrix_from_pheno_scores(
        test_subject_IDs,   # list of IDs for test set (TCD)
        dev_subject_IDs,    # list of IDs for development set
        pheno_csv_path)

# If no pheno data is used, pheno_affinity_matrix is set to ones 
else:
    pheno_test_to_dev = np.ones((len(test_subject_IDs), len(dev_subject_IDs)), dtype=np.float32)
    pheno_affinity_matrix = np.ones((len(dev_y_true), len(dev_y_true)), dtype=np.float32)

# Inductive test evaluation
test_results = test_gnn_inductive(
    args,
    final_model,
    dev_features=node_ftr,                          # Final dev features after feature selection
    selected_idx=selected_idx,                      # Indices of selected features
    pheno_affinity_dev=pheno_affinity_matrix,       # Dev × dev phenotypic similarity
    test_features=test_features,                    # Test SC features
    test_labels=test_y_true,
    pheno_affinity_test_to_dev=pheno_test_to_dev,   # Test × dev phenotypic similarity
    device=device,
    pos_class_index=0,
)

# Merge test metrics into final model results
final_model_results.update(test_results)

print("\r\n========================== Finish ==========================")

print(f"=> Average train accuracy in {n_folds}-fold CV: "
      f"{mean_results['train_accuracy'][0]:.4f} ± {mean_results['train_accuracy'][1]:.4f}")

print(f"=> Average validation accuracy in {n_folds}-fold CV: "
      f"{mean_results['val_accuracy'][0]:.4f} ± {mean_results['val_accuracy'][1]:.4f}")

print(f"=> Average val AUC in {n_folds}-fold CV: "
      f"{mean_results['auc'][0]:.4f} ± {mean_results['auc'][1]:.4f}")

print(f"=> Average val F1-score {mean_results['f1_score'][0]:.4f} ± {mean_results['f1_score'][1]:.4f}, "
      f"Recall {mean_results['recall'][0]:.4f} ± {mean_results['recall'][1]:.4f}, "
      f"Specificity {mean_results['specificity'][0]:.4f} ± {mean_results['specificity'][1]:.4f}, "
      f"Precision {mean_results['precision'][0]:.4f} ± {mean_results['precision'][1]:.4f}, "
      f"NPV {mean_results['npv'][0]:.4f} ± {mean_results['npv'][1]:.4f}")


# Save results (CV, final model, args, and per-subject predictions)
pd.set_option('display.precision', 4)

df_folds = pd.DataFrame(fold_results)   # One row per CV fold

# CV mean and standard deviation
cv_mean = {f"cv_mean_{k}": v[0] for k, v in mean_results.items()}
cv_std  = {f"cv_std_{k}":  v[1] for k, v in mean_results.items()}

# Args (hyperparameters) used for this run
args_dict = vars(args)
df_args = pd.DataFrame({
    "parameter": list(args_dict.keys()),
    "value": list(args_dict.values())
})

args_prefixed = {f"arg_{k}": v for k, v in args_dict.items()}

# Separate out the list-valued items from final_model_results
list_keys = ['y_true', 'y_pred', 'prob_pos']
final_metrics_only = {k: v for k, v in final_model_results.items() if k not in list_keys}

# Summary row (CV stats + final metrics + args)
summary_row = {
    "fold": "SUMMARY",
    **cv_mean,
    **cv_std,
    **{f"final_{k}": v for k, v in final_metrics_only.items()},
    **args_prefixed,
}

df_all = pd.concat([df_folds, pd.DataFrame([summary_row])], ignore_index=True)

# Per-test-subject predictions
test_pred_df = pd.DataFrame({
    "SUB_ID": test_subject_IDs,
    "y_true": final_model_results["y_true"],
    "y_pred": final_model_results["y_pred"],
    "prob_pos": final_model_results["prob_pos"],
})

csv_path = os.path.join(results_dir, "GNN_results.xlsx")
with pd.ExcelWriter(csv_path) as writer:
    df_all.to_excel(writer, sheet_name="cv_and_summary", index=False)
    test_pred_df.to_excel(writer, sheet_name="test_predictions", index=False)
    df_args.to_excel(writer, sheet_name="args", index=False)

print("Saved results to CSV:", csv_path)

