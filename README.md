# GNN for ASD classification using structural connectivity
A population-based GNN approach for ASD classification using diffusion MRI-based structural connectivity from the ABIDE-II database.

This project implements a structural connectivity (SC) pipeline and population-based GNN for ASD classification using diffusion MRI from the ABIDE-II dataset. It includes preprocessing scripts, graph construction, a flexible GNN architecture, evaluation tools, and scripts for cross-validation and inductive testing.

It includes:
- Preprocessing + tractography to generate FA-weighted structural connectivity (SC) matrices
- Construction of population graphs (RBF / RBF+PS)
- GNN training with cross-validation
- Hyperparameter optimization
- Inductive test evaluation

# Folder Structure & Script Descriptions

## ABIDE_preprocessing/

Scripts for converting raw diffusion MRI data into FA-weighted SC matrices.

### aal/ 
Contains the AAL2 atlas used for region-based parcellation.

### combine_csv.py
Merges subject CSV files into a single phenotype file.

### preproc_and_tractography.sh
Full preprocessing pipeline:
- DWI denoising and Gibbs removal
- Eddy correction
- Brain masking
- Fiber orientation estimation (CSD)
- Anatomically constrained tractography (ACT)
- Streamline-to-region assignment
- Creation of FA-weighted connectivity matrices

This generates the SC features used by the GNN.

### run_all_subjects.sh
Wrapper script for running the preprocessing + tractography pipeline on all subjects.

### show_connectivitymatrix.py
Utility for plotting individual or group-average SC matrices and inspecting the strongest connections.


## GNN_model/
Implements the population GNN, feature selection, harmonization, training, testing, and hyperparameter optimization.

### MAIN.py

Main entry point:

- Load SC features & phenotypic data
- Apply optional ComBat harmonization
- Perform feature selection
- Build population graph
- Train a GNN using cross-validation
- Retrain final model on all development subjects
- Run inductive testing on held-out subjects
- Save metrics + plots

### GNNModel.py
Defines the GNN architecture:
- GraphConv (default), GAT, GINE, or ChebConv
- Edge + feature dropout
- Sum-based message passing
- LayerNorm + ReLU / PReLU
- MLP classifier head

Handles the forward pass and parameter initialization.

### NEW_hyperparameter_optuna.py
Runs Optuna hyperparameter optimization:
- Samples hyperparameters (dropout, LR, threshold, etc.)
- Performs full cross-validation inside each trial
- Selects the best-performing configuration.

### args.py
Stores all configurable hyperparameters:
- GNN architecture
- Training settings
- Graph construction options
- Dataset splitting
- Harmonization and batching options

### data_processing.py
Responsible for preparing the dataset:
- Load SC matrices and flatten them
- Apply feature selection (Ridge + RFE)
- Compute RBF similarity
- Compute phenotypic similarity (sex + age)
- Build adjacency matrices
- Apply ComBat harmonization
- Perform stratified development/test split

### metrics.py
Implements evaluation metrics:
- Accuracy, AUC, precision, recall, F1, specificity, NPV

### train_and_test.py

Contains training and evaluation logic:
- 5-fold cross-validation with early stopping
- Final model refitting on the full development set
- Inductive testing (test subjects added one-by-one into the population graph)
- Saves results, curves, and fold-level summaries.

### utils.py

Helper functions for:
- Plotting training curves
- Summarizing cross-validation metrics
- Formatting mean ± std statistics

.gitignore

Excludes large files, temporary artifacts, and local environment files from version control.

🔍 Short Summary (for the top of README)

