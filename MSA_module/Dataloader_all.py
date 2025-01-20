import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceParatopeDataset(Dataset):
    """
    Dataset class where MSA sequences are the input, and paratope, secondary structure (ss),
    SASA (sasa), SAPT, and edge data are the targets.
    """
    def __init__(self, data_file, sequence_file, edge_file, max_len=1200):
        """
        Args:
            data_file (str): Path to the .npz file containing ss, sasa, pt, and sapt data.
            sequence_file (str): Path to the .npz file containing sequence data.
            edge_file (str): Path to the .npz file containing edge data.
            max_len (int): Maximum sequence length for padding/truncation.
        """
        # Load sequence and data files
        self.sequence_data = np.load(sequence_file, allow_pickle=True)["sequences"]
        data = np.load(data_file, allow_pickle=True)
        self.ss_data = data["ss_data"]
        self.sasa_data = data["sasa_data"]
        self.pt_data = data["pt_data"]
        self.sapt_data = data["sapt"]
        self.identifiers = data["identifiers"]

        # Load edge data
        edges = np.load(edge_file, allow_pickle=True)
        self.edge_identifiers = edges["identifiers"]
        self.edge_data = edges["edges"]

        self.max_len = max_len

        # Ensure identifiers are aligned
        if not np.array_equal(self.identifiers, self.edge_identifiers):
            raise ValueError("Identifiers in data and edges are not aligned.")

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        # Load sequence and pad/truncate to (100, max_len)
        sequence = self.sequence_data[idx]
        if sequence.shape[1] < self.max_len:
            sequence = np.pad(sequence, ((0, 0), (0, self.max_len - sequence.shape[1])), constant_values=-1)
        sequence = sequence[:, :self.max_len]

        if sequence.shape[0] < 100:
            sequence = np.pad(sequence, ((0, 100 - sequence.shape[0]), (0, 0)), constant_values=-1)
        sequence = sequence[:100, :]

        # Load the corresponding secondary structure, SASA, and paratope data
        ss = self.ss_data[idx]
        sasa = self.sasa_data[idx]
        pt = self.pt_data[idx]

        # Pad/truncate ss, sasa, and pt to max_len
        ss = (list(ss) + [-1] * self.max_len)[:self.max_len]
        sasa = (list(sasa) + [-1.0] * self.max_len)[:self.max_len]
        pt = (list(pt) + [-1] * self.max_len)[:self.max_len]

        # Load SAPT as a scalar value
        sapt = self.sapt_data[idx]

        # Load edges for the current index
        edges = self.edge_data[idx]

        # Convert all to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        ss_tensor = torch.tensor(ss, dtype=torch.long)
        sasa_tensor = torch.tensor(sasa, dtype=torch.float)
        pt_tensor = torch.tensor(pt, dtype=torch.long)
        sapt_tensor = torch.tensor(sapt, dtype=torch.float)
        edges_tensor = torch.tensor(edges, dtype=torch.long)  # Assuming edges are integers

        # Return all data as a tuple
        return sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor, edges_tensor


