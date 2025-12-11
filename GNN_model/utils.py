import os
import csv
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
    
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch_geometric.utils import dense_to_sparse


# Get phenotype values for a list of subject IDs
def collect_pheno_scores_in_dict(pheno_csv_path, subject_ids, label_column):
    """
    Return a dictionary mapping subject IDs to a phenotype value.

    Inputs:
        pheno_csv_path: path to phenotype CSV
        subject_ids: list of subject IDs to look up
        label_column: phenotype column to extract (e.g. 'SITE_ID')

    Output:
        Dict(subject_id : phenotype_value)
    """

    pheno_dict = {}

    with open(pheno_csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_ids:
                pheno_dict[row['SUB_ID']] = row[label_column]

    return pheno_dict


def collect_subjects_of_sites(data_folder, site_names, matrix_filename="conn_FAmean.csv"):
    """ 
    Collect subject IDs and SC matrix file paths for a set of sites.

    Inputs:
        data_folder: root data directory
        site_names: list of site folder names
        matrix_filename: SC matrix file to look for

    Outputs:
        subject_ids: array of found subject IDs
        file_paths: paths to their SC matrix files
    """

    subject_ids = []
    file_paths = []

    for site in site_names:
        site_dir = os.path.join(data_folder, site)
        for subj in sorted(os.listdir(site_dir)):
            subj_dir = os.path.join(site_dir, subj)
            sc_path = os.path.join(subj_dir, "session_1", "dti_1", matrix_filename)
            if os.path.isfile(sc_path):
                subject_ids.append(subj)
                file_paths.append(sc_path)
    
    return np.array(subject_ids), file_paths


def load_matrices_and_flatten(csv_paths):
    """ 
    Load SC matrices from CSV files and flatten them.
    Each matrix is converted to a 1D feature vector by taking the
    upper-triangular part (excluding the diagonal).

    Input:
        csv_paths: list of file paths to SC matrices

    Output:
        features: array of shape (subjects, features)
    """

    features = []

    for path in csv_paths:
        conn_matrix = pd.read_csv(path, header=None).values

        if conn_matrix.shape[0] != conn_matrix.shape[1]:
            raise ValueError(f"Non-square matrix at {path}: {conn_matrix.shape}")

        upper_triangle = np.triu_indices(conn_matrix.shape[0], k=1)
        feat_vector = conn_matrix[upper_triangle].astype(np.float32)
        features.append(feat_vector)

    features = np.vstack(features)

    return features


def check_graph_connectivity(edge_index, num_nodes, plot=True, figsize=(8,6)):
    """
    Check for the connectivity of the graph and identify isolated nodes.

    Inputs:
        edge_index: graph edges
        num_nodes: number of nodes in the graph
        plot: whether to plot the graph layout
        figsize: size of the plot

    Outputs:
        isolated: list of node indices with degree 0
        degrees: degree of every node
    """

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    # Compute degrees
    degrees = np.zeros(num_nodes, dtype=int)
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src != dst:          # Ignore self-loops
            degrees[src] += 1
            degrees[dst] += 1
    
    # Identify isolated nodes
    isolated = np.where(degrees == 0)[0].tolist()

    print("\nGraph connectivity:")
    print(f"Number of nodes           : {num_nodes}")
    print(f"Number of edges           : {edge_index.shape[1]}")
    print(f"Isolated nodes            : {len(isolated)}")

    if len(isolated) > 0:
        print("Isolated node indices:", isolated)
    else:
        print("No isolated nodes detected.")

    # Optional plot
    if plot:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=50, width=0.3)
        plt.title("Population Graph")
        plt.show()

    return isolated, degrees


def feature_selection(features, y_true, train_ind, n_reduced):
    """
    Perform feature selection using a ridge classifier + RFE.

    Inputs:
        features: full feature matrix (subjects x features)
        y_true: labels for all subjects
        train_ind: indices used for fitting the selector
        n_reduced: number of features to keep after selection

    Outputs:
        x_data: reduced feature matrix (subjects x n_reduced)
        selected_idx: indices of the selected features
    """

    # Iteratively removes features until n_reduced remain
    estimator = RidgeClassifier()
    selector = RFE(estimator, 
                   n_features_to_select=n_reduced, 
                   step=100,        # Remove 100 features per iteration
                   verbose=0)

    # Fit the selector only on the training subjects
    featureX = features[train_ind, :]
    featureY = y_true[train_ind]
    selector = selector.fit(featureX, featureY.ravel())

    # Transform the full dataset to the reduced feature space
    x_data = selector.transform(features)

    # Indices of selected features in the original vector
    selected_idx = selector.get_support(indices=True)

    return x_data, selected_idx


def create_affinity_matrix_from_pheno_scores(subject_list, pheno_csv_path):
    """
    Build a phenotypic similarity matrix based on sex and age.

    Inputs:
        subject_list: list of subject IDs
        pheno_csv_path: path to phenotype CSV file

    Output:
        pheno_affinity_matrix: (N x N) matrix with similarity scores in [0, 1]
    """

    num_subjects = len(subject_list)
    pheno_affinity_matrix = np.zeros((num_subjects, num_subjects), dtype=np.float32)

    # Get each phenotypic score as a dictionary
    sex_dict = collect_pheno_scores_in_dict(pheno_csv_path, subject_list, 'SEX')
    age_dict = collect_pheno_scores_in_dict(pheno_csv_path, subject_list, 'AGE_AT_SCAN')

    all_sex = []
    all_age = []

    # Convert to aligned arrays in the same subject order
    for subj in subject_list:
        sex = float(sex_dict[subj])
        all_sex.append(sex)

        age = float(age_dict[subj])
        all_age.append(age)

    all_sex  = np.array(all_sex,  dtype=float)
    all_age  = np.array(all_age,  dtype=float)

    # Standard scale age
    scaled_age = StandardScaler().fit_transform(all_age.reshape(-1, 1)).ravel()

    # Compute pairwise phenotypic similarity
    for i in range(num_subjects):
        for j in range(num_subjects):
            same_sex = 1.0 if all_sex[i] == all_sex[j] else 0.0
            age_term = 1.0 / (1.0 + abs(scaled_age[i] - scaled_age[j]))
            pheno_affinity_matrix[i, j] = (same_sex + age_term) / 2     # Maximum value in pheno graph is 1

    return pheno_affinity_matrix


# def get_static_affinity_adj(features, subject_list):
#     pd_affinity = create_affinity_matrix_from_pheno_scores(['SEX', 'SITE_ID', 'AGE_AT_SCAN'], subject_list)
#     distv = distance.pdist(features, metric='correlation') 
#     dist = distance.squareform(distv)  
#     sigma = np.mean(dist)
#     feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
#     adj = pd_affinity * feature_sim  

#     return adj
    

def compute_total_affinity_graph(node_ftr, pheno_aff_matrix, graph_method, 
                                 affinity_threshold=None, k=None):
    """
    Build the final population graph by combining RBF with optional phenotypic similarity.

    Inputs:
        node_ftr: matrix of node features (subjects x features)
        pheno_aff_matrix: phenotypic similarity matrix (or None)
        graph_method: 'aff_threshold' or 'kNN'
        affinity_threshold: threshold for pruning edges
        k: number of neighbours to keep

    Outputs:
        edge_index: edge list
        edge_attr: edge weights for each edge
        final_graph: affinity matrix after pruning
    """

    # Compute pairwise correlation distance between subjects
    distv = distance.pdist(node_ftr, metric='correlation')
    dist = distance.squareform(distv)

    # RBF similarity from correlation distance
    sigma = np.mean(dist)
    RBF_matrix = np.exp(- dist ** 2 / (2 * sigma ** 2))

    # Combine imaging + phenotypic similarity if used
    if pheno_aff_matrix is not None:
        final_graph = pheno_aff_matrix * RBF_matrix
    else:
        final_graph = RBF_matrix

    num_nodes = final_graph.shape[0]

    # Check for graph construction method
    if graph_method == "aff_threshold":
        if affinity_threshold is None:
            raise ValueError("affinity_threshold must be provided for aff_threshold method.")

        # Keep edges with similarity above threshold
        final_graph = np.where(final_graph >= affinity_threshold, final_graph, 0)

        # Report edges
        num_edges = np.count_nonzero(final_graph > 0) // 2
        # print(f"[Graph] Threshold = {affinity_threshold}, edges = {num_edges}")

    elif graph_method == "kNN":
        if k is None:
            raise ValueError("k must be provided for kNN graph method.")

        # Remove self-connections
        np.fill_diagonal(final_graph, 0)

        # Build kNN adjacency matrix
        knn_graph = np.zeros_like(final_graph)

        for i in range(num_nodes):
            # Indices of top-k neighbors
            nn_idx = np.argsort(final_graph[i])[-k:]
            knn_graph[i, nn_idx] = final_graph[i, nn_idx]

        # Make symmetric (undirected graph)
        final_graph = np.maximum(knn_graph, knn_graph.T)

        num_edges = np.count_nonzero(final_graph > 0) // 2
        print(f"[Graph] kNN (k={k}), edges = {num_edges}")

    else:
        raise ValueError(f"Unknown graph_method: {graph_method}")

    final_graph = torch.tensor(final_graph, dtype=torch.float32)
    edge_index, edge_attr = dense_to_sparse(final_graph)
    edge_attr = edge_attr.unsqueeze(-1)

    return edge_index, edge_attr, final_graph


def cross_affinity_matrix_from_pheno_scores(test_subject_IDs, dev_subject_IDs, pheno_csv_path):
    """
    Compute phenotypic similarity between test and development subjects
    using the same formula as in the development graph (sex + scaled age).

    Inputs:
        test_subject_IDs: list of subject IDs in the test set
        dev_subject_IDs: list of subject IDs in the development set
        pheno_csv_path: path to the phenotype CSV file

    Output:
        cross_affinity: matrix of shape (N_test x N_dev) containing
                         phenotypic similarity values in [0, 1]
    """
    
    test_subject_IDs = list(test_subject_IDs)
    dev_subject_IDs = list(dev_subject_IDs)

    num_test = len(test_subject_IDs)
    num_dev = len(dev_subject_IDs)
    cross_affinity = np.zeros((num_test, num_dev), dtype=np.float32)

    # Load sex and age for all subjects
    sex_dict = collect_pheno_scores_in_dict(pheno_csv_path,
                                            test_subject_IDs + dev_subject_IDs,
                                            'SEX')
    age_dict = collect_pheno_scores_in_dict(pheno_csv_path,
                                            test_subject_IDs + dev_subject_IDs,
                                            'AGE_AT_SCAN')

    # Extract arrays in consistent order
    sex_test = np.array([float(sex_dict[s]) for s in test_subject_IDs], dtype=float)
    sex_dev  = np.array([float(sex_dict[s]) for s in dev_subject_IDs], dtype=float)

    age_test = np.array([float(age_dict[s]) for s in test_subject_IDs], dtype=float)
    age_dev  = np.array([float(age_dict[s]) for s in dev_subject_IDs], dtype=float)

    # Standard-scale age using only the development set
    scaler = StandardScaler().fit(age_dev.reshape(-1, 1))
    age_dev_scaled  = scaler.transform(age_dev.reshape(-1, 1)).ravel()
    age_test_scaled = scaler.transform(age_test.reshape(-1, 1)).ravel()

    # Compute phenotypic similarity (same formula as dev-dev)
    for i in range(num_test):
        for j in range(num_dev):
            same_sex = 1.0 if sex_test[i] == sex_dev[j] else 0.0
            age_term = 1.0 / (1.0 + abs(age_test_scaled[i] - age_dev_scaled[j]))

            # Match development graph computation
            cross_affinity[i, j] = (same_sex + age_term) / 2

    return cross_affinity
