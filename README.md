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
Merges dataset CSV files into a single phenotype file.

### preproc_and_tractography.sh
Full preprocessing pipeline:
- DWI denoising and Gibbs removal
- Eddy correction
- Brain masking
- Fiber orientation estimation and constrained spherical deconvolution (CSD)
- Anatomically constrained tractography (ACT)
- Creation of FA-weighted connectivity matrices

This generates the SC features used by the GNN.

### run_all_subjects.sh
Wrapper script for running the preprocessing + tractography pipeline on all subjects.

### show_connectivitymatrix.py
Script for plotting individual or group-average SC matrices and inspecting connections.


## GNN_model/
Implements the population GNN, feature selection, harmonization, training, testing, and hyperparameter optimization.

### MAIN.py
Main script, that runs the whole GNN pipeline:
- Loads structural connectivity and phenotype data.
- Builds development/test splits and applies optional ComBat harmonization.
- Runs 5-fold cross-validation to tune hyperparameters.
- Trains the final GNN on all development subjects.
- Performs inductive testing on unseen subjects and saves all results.

### args.py
Stores all the hyperparameters.

### data_processing.py
Handles dataset preparation:
- Loads and flattens SC matrices.
- Applies feature selection (Ridge + RFE).
- Computes RBF + phenotypic similarity.
- Builds adjacency matrices and applies ComBat harmonization.
- Creates stratified development/test splits.

### train_and_test.py
Contains training and evaluation:
- 5-fold cross-validation with early stopping
- Final model refitting on the full development set
- Inductive testing (test subjects added one-by-one into the population graph)
- Saves results, curves, and fold-level summaries.

### GNNModel.py
Defines the GNN architecture:
- GraphConv (default), GAT, GINE, or ChebConv
- Edge + feature dropout
- Message passing
- LayerNorm + ReLU / PReLU
- MLP classifier head

Handles the forward pass and parameter initialization.

### metrics.py
Computed evaluation metrics:
- Accuracy, AUC, precision, recall, F1, specificity, NPV

### utils.py
Provides helper functions for:
- Loading phenotype values and computing phenotypic similarity (sex + age).
- Loading SC matrices and flattening them into feature vectors.
- Building population graphs.
- Feature selection (Ridge + RFE).

### hyperparameter_optuna.py
Runs Optuna hyperparameter optimization:
- Define search spaces for hyperparameters (dropout, LR, threshold, etc.)
- Performs full cross-validation inside each trial
- Saves the trial results to CSV






