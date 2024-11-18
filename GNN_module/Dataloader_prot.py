import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix

class ProteinGraphDataset(Dataset):
    def __init__(self, residue_features_list, adj_matrix_list):
        """
        Args:
        residue_features_list: List of residue feature arrays (each array: [num_residues, feature_dim])
        adj_matrix_list: List of adjacency matrices (each matrix: [num_residues, num_residues])
        """
        self.residue_features_list = residue_features_list
        self.adj_matrix_list = adj_matrix_list

    def __len__(self):
        return len(self.residue_features_list)

    def __getitem__(self, idx):
        residue_features = torch.tensor(self.residue_features_list[idx], dtype=torch.float)
        adj_matrix = csr_matrix(self.adj_matrix_list[idx])
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)

        data = Data(x=residue_features, edge_index=edge_index)
        return data

# Function to create a dataloader
def create_dataloader(residue_features_list, adj_matrix_list, batch_size=32, shuffle=True):
    dataset = ProteinGraphDataset(residue_features_list, adj_matrix_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

