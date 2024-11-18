import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse import csr_matrix

# Define the GAT model with adjacency matrix update
class DynamicGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, threshold=0.5):
        super(DynamicGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        self.threshold = threshold  # Threshold for creating new edges based on learned features

    def forward(self, data):
        # Initial node features and edge index (adjacency matrix)
        x, edge_index = data.x, data.edge_index

        # First GAT layer + activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        # Dynamically update the adjacency matrix based on learned features
        updated_adj_matrix = self.update_adjacency_matrix(x)

        # Recompute the edge index from the updated adjacency matrix
        edge_index, _ = from_scipy_sparse_matrix(updated_adj_matrix)

        # Second GAT layer using the updated adjacency matrix
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def update_adjacency_matrix(self, node_features):
        """
        Update the adjacency matrix based on node feature similarities (e.g., distance or similarity).
        For simplicity, we use Euclidean distance between node features and a threshold.
        """
        # Calculate pairwise distances (or similarities) between node features
        num_nodes = node_features.size(0)
        node_features = node_features.detach().cpu().numpy()
        distances = np.linalg.norm(node_features[:, None] - node_features, axis=2)

        # Create a new adjacency matrix where edges are formed if distance < threshold
        adj_matrix = (distances < self.threshold).astype(int)

        # Convert to sparse matrix for efficiency
        adj_matrix = csr_matrix(adj_matrix)

        return adj_matrix


