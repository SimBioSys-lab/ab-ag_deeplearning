import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class SequenceSASADataset(Dataset):
    """
    Dataset class where MSA sequences are the input and SASA values are the target.
    """
    def __init__(self, sequence_file, sasa_file, max_len=1200):
        # Load the sequence data from npz file
        self.sequence_data = np.load(sequence_file)['sequences']
        # Load the SASA data from a CSV file
        self.sasa_data = pd.read_csv(sasa_file, header=None, index_col=0)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        # Get the sequence data and pad/truncate to (100, max_len)
        sequence = self.sequence_data[idx]
        if sequence.shape[1] < self.max_len:
            # Pad each row to max_len
            sequence = np.pad(sequence, ((0, 0), (0, self.max_len - sequence.shape[1])), constant_values=-1)
        sequence = sequence[:, :self.max_len]  # Truncate to max_len columns if needed

        if sequence.shape[0] < 100:
            # Pad rows to reach 100 rows
            sequence = np.pad(sequence, ((0, 100 - sequence.shape[0]), (0, 0)), constant_values=-1)
        sequence = sequence[:100, :]  # Truncate to 100 rows if needed

        # Get the SASA values corresponding to the same index, excluding the identifier
        sasa = self.sasa_data.iloc[idx, :].dropna().astype(float).tolist()
        sasa = (sasa + [-1] * self.max_len)[:self.max_len]

        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        sasa_tensor = torch.tensor(sasa, dtype=torch.float32)

        return sequence_tensor, sasa_tensor


class SequenceSecondaryStructureDataset(Dataset):
    """
    Dataset class where MSA sequences are the input and secondary structure data are the target.
    """
    def __init__(self, sequence_file, ss_file, max_len=1200):
        # Load the sequence data from npz file
        self.sequence_data = np.load(sequence_file)['sequences']
        # Load the secondary structure data from CSV
        self.ss_data = pd.read_csv(ss_file, header=None, index_col=0)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        # Get the sequence data and pad/truncate to (100, max_len)
        sequence = self.sequence_data[idx]
        if sequence.shape[1] < self.max_len:
            sequence = np.pad(sequence, ((0, 0), (0, self.max_len - sequence.shape[1])), constant_values=-1)
        sequence = sequence[:, :self.max_len]

        if sequence.shape[0] < 100:
            sequence = np.pad(sequence, ((0, 100 - sequence.shape[0]), (0, 0)), constant_values=-1)
        sequence = sequence[:100, :]


        # Get the secondary structure values corresponding to the same index
        secondary_structure = self.ss_data.iloc[idx, :].dropna().astype(int).tolist()
        secondary_structure = (secondary_structure + [-1] * self.max_len)[:self.max_len]

        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        ss_tensor = torch.tensor(secondary_structure, dtype=torch.long)

        return sequence_tensor, ss_tensor


class SequenceParatopeDataset(Dataset):
    """
    Dataset class where MSA sequences are the input and paratope data are the target.
    """
    def __init__(self, sequence_file, pt_file, max_len=1200):
        # Load the sequence data from npz file
        self.sequence_data = np.load(sequence_file)['sequences']
        # Load the paratope data from CSV, where each row represents a paratope mask
        self.pt_data = pd.read_csv(pt_file, header=None, index_col=0)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        # Get the sequence data and pad/truncate to (100, max_len)
        sequence = self.sequence_data[idx]
        if sequence.shape[1] < self.max_len:
            sequence = np.pad(sequence, ((0, 0), (0, self.max_len - sequence.shape[1])), constant_values=-1)
        sequence = sequence[:, :self.max_len]

        if sequence.shape[0] < 100:
            sequence = np.pad(sequence, ((0, 100 - sequence.shape[0]), (0, 0)), constant_values=-1)
        sequence = sequence[:100, :]

        # Get the secondary structure values corresponding to the same index
        paratope = self.pt_data.iloc[idx, :].dropna().astype(int).tolist()
        paratope = (paratope + [-1] * self.max_len)[:self.max_len]

        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        pt_tensor = torch.tensor(paratope, dtype=torch.long)

        return sequence_tensor, pt_tensor

class SequenceSAPTDataset(Dataset):
    """
    Dataset class where MSA sequences are the input and a single SAPT value is the target.
    """
    def __init__(self, sequence_file, sapt_file, max_len=1200):
        # Load the sequence data from the npz file
        self.sequence_data = np.load(sequence_file)['sequences']
        # Load the SAPT data from a CSV file
        self.sapt_data = pd.read_csv(sapt_file, header=None, index_col=0)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        # Get the sequence data and pad/truncate to (100, max_len)
        sequence = self.sequence_data[idx]
        if sequence.shape[1] < self.max_len:
            # Pad each row to max_len
            sequence = np.pad(sequence, ((0, 0), (0, self.max_len - sequence.shape[1])), constant_values=-1)
        sequence = sequence[:, :self.max_len]  # Truncate to max_len columns if needed

        if sequence.shape[0] < 100:
            # Pad rows to reach 100 rows
            sequence = np.pad(sequence, ((0, 100 - sequence.shape[0]), (0, 0)), constant_values=-1)
        sequence = sequence[:100, :]  # Truncate to 100 rows if needed

        # Get the single SAPT value corresponding to the same index
        sapt = self.sapt_data.iloc[idx, 0]  # Assume SAPT is stored as the first column for each sequence
        sapt = float(sapt)  # Ensure it's a float

        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        sapt_tensor = torch.tensor(sapt, dtype=torch.float32)

        return sequence_tensor, sapt_tensor

