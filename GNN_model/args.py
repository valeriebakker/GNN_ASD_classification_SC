class Args:
    def __init__(self):
        self.dropout = 0.028		     # Fraction of node features randomly set to 0 during training to reduce overfitting.
        self.edge_dropout = 0.117	     # Fraction of edges randomly dropped during training to regularize the graph structure.
        self.epochs = 500                # Maximum number of training epochs.
        self.weight_decay = 1.55e-5       # L2 regularization on the weights inside the optimizer. Helps prevent very large weights.
        self.lr = 	0.0002               # Learning rate (step size) for the optimizer.
        self.node_ftr_dim = 300          # Number of features kept after feature selection per subject (reduced from the full SC vector).
        self.lg = 2                      # Number of GNN layers
        self.hiddenU = [8,8]             # Hidden dimensionality for each GNN layer. Must match the number of layers (lg).
        self.cls_hidden = 64             # Hidden layer size of the MLP classifier after GNN layers.
        self.patience = 50               # Amount of epochs with no stable increase of performance before early stopping
        self.model = "GraphConv"         # Type of GNN layer used
        self.ckpt_path = None            # Folder name for checkpoints
        self.num_classes = 2             # Number of output classes (ASD vs. TDC).
        self.split_mode = 'mixed'        # How to split development and test sets: 'mixed' or 'loso' (leave-one-site-out).

        if self.split_mode == 'mixed':
            self.test_percentage = 0.2              # Percentage of test split

        self.use_combat = True                      # Whether to apply ComBat harmonization to SC features.
        self.use_pheno_data = True                  # Whether to include phenotypic similarity in the edge weights.
        self.graph_method = "aff_threshold"         # Graph construction method: threshold edges or use kNN.

        if self.graph_method == "aff_threshold":
            self.affinity_threshold = 0.524         # Similarity threshold below which edges are pruned.
            self.k = None
        elif self.graph_method == "kNN":
            self.k = 15                             # Number of nearest neighbours to connect to each node.
            self.affinity_threshold = None

        self.use_batching = False                   # Whether to use mini-batch sampling (neighbourhood sampling) instead of full-graph training.
        if self.use_batching == True:
            self.batch_size = 8                     # Batch size when sampling subgraphs.
            self.num_neighbors = [25, 10, 8, 4]     # Number of neighbours sampled at each GNN layer during batching.

        self.activation = 'relu'                    # Activation function used in GNN layers and MLP classifier.
        if self.activation == 'prelu':
            self.prelu_unit = 0.25                  # Initial parameter for PReLU activation.