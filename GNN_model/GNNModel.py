import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, ChebConv, GraphConv, LayerNorm
from torch_geometric.utils import dropout_edge

class GNNmodel(nn.Module):
    def __init__(self, num_features, edgenet_input_dim, out_dim, args):
        super().__init__()

        # Read hyperparameters from args
        self.lg = args.lg
        self.hidden = args.hiddenU
        self.dropout = args.dropout
        self.edge_dropout = args.edge_dropout
        self.current_epoch = 0
        self.max_epochs = args.epochs 

        # Activation function
        self.use_prelu = (getattr(args, "activation", "prelu").lower() == "prelu")
        if self.use_prelu:
            # One PReLU unit per GNN layer
            self.acts = nn.ModuleList([nn.PReLU(init=getattr(args, "prelu_init", 0.25)) for _ in range(self.lg)])
        else:
            self.acts = None 
        
        torch.manual_seed(1234567)

        # Initialize GNN layers and layer normalizations
        self.gconvs = nn.ModuleList()       # GNN layer
        self.norms = nn.ModuleList()        # LayerNorm after each GNN layer

        for i in range(self.lg):
            # Input dimension: raw features for first layer, hidden dim for others
            in_dim = num_features if i == 0 else self.hidden[i - 1]
            out_dim = self.hidden[i]

            # Create GNN layer and corresponding normalization layer
            self.gconvs.append(self.build_conv_model(in_dim, out_dim, args))
            self.norms.append(LayerNorm(out_dim))
        
        # Output dimension of final GNN layer becomes input to classifier MLP   
        cls_input_dim = self.hidden[self.lg - 1]

        # Classification head (MLP)
        # Maps the final node embedding to ASD vs. TDC logits
        self.cls = nn.Sequential(
            nn.Linear(cls_input_dim, args.cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            LayerNorm(args.cls_hidden),
            nn.Linear(args.cls_hidden, args.num_classes))
        
        # Initialization of model parameters
        self.model_init()


    def build_conv_model(self, input_dim, hidden_dim, args):
        """Return the requested convolution layer."""
        if args.model == "Cheb":
            return ChebConv(input_dim, hidden_dim, K=3, normalization="sym")

        if args.model == "GIN":
            nn_gin = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            return GINEConv(nn_gin, edge_dim=1)

        # GAT uses learned attention weights to compute the importance of each neighbour.
        if args.model == "GAT":
            return GATv2Conv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=1,
                dropout=self.dropout,
                edge_dim=1,
                share_weights=True
            )

        # GraphConv weights neighbour features and sums them (by using "add").
        if args.model == "GraphConv":
            return GraphConv(input_dim, hidden_dim, aggr="add")

        raise ValueError(f"Unknown model type: {args.model}")


    # Parameter initialization
    def model_init(self):
        # Initialize parameters of all submodules
        for m in self.modules():

            # Initialize linear layers using Kaiming normal
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # Initialize the MLP inside a GINEConv layer
            if isinstance(m, GINEConv):
                for layer in m.nn:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)


    # Forward pass of the model
    def forward(self, features, edge_index, edgenet_input):
        try:
            # Apply edge dropout during training
            edge_index, edge_mask = dropout_edge(
                edge_index, p=self.edge_dropout, training=self.training)
            
            # Filter edge attributes so they match the dropped edges
            edgenet_input = edgenet_input[edge_mask] if edgenet_input is not None else None

        except Exception as e:
            raise ValueError(f"Error in dropout_edge: {e}")

        # Edge weights used by convolution layers (may be None depending on model)
        edge_weights = edgenet_input

        # Apply dropout to node features
        h = F.dropout(features, p=self.dropout, training=self.training)

        # Message-passing layers
        for i in range(self.lg):

            # Different conv layers expect different edge-weight formats
            if isinstance(self.gconvs[i], (GraphConv, ChebConv)):
                # GraphConv and ChebConv expect a 1D tensor for edge weights
                h = self.gconvs[i](h, edge_index, edge_weights.view(-1))
            else:
                # GAT/GINEConv take edge weights directly
                h = self.gconvs[i](h, edge_index, edge_weights)

            # Layer normalization
            h = self.norms[i](h)

            # Check for PReLU or ReLU for activation
            if self.use_prelu:
                h = self.acts[i](h)
            else:
                h = F.relu(h, inplace=False)

        # Final classification head
        logits = self.cls(h)

        return logits