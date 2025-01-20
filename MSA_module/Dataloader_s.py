import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceParatopeDataset(Dataset):
    """
    Dataset class where MSA sequences are the input, and paratope, secondary structure (ss),
    SASA (sasa), SAPT, and edge data are the targets.
    """
    def __init__(self, abdata_file, agdata_file, absequence_file, agsequence_file, abedge_file, agedge_file, abmax_len=1300, agmax_len=2500):
        """
        Args:
            abdata_file (str): Path to the .npz file containing antibody data.
            agdata_file (str): Path to the .npz file containing antigen data.
            absequence_file (str): Path to the .npz file containing antibody sequence data.
            agsequence_file (str): Path to the .npz file containing antigen sequence data.
            abedge_file (str): Path to the .npz file containing antibody edge data.
            agedge_file (str): Path to the .npz file containing antigen edge data.
            abmax_len (int): Maximum sequence length for antibody data.
            agmax_len (int): Maximum sequence length for antigen data.
        """
        self.absequence_data = np.load(absequence_file, allow_pickle=True)
        self.abpt_data = np.load(abdata_file, allow_pickle=True)
        self.abedge_data = np.load(abedge_file, allow_pickle=True)
        self.agsequence_data = np.load(agsequence_file, allow_pickle=True)
        self.agpt_data = np.load(agdata_file, allow_pickle=True)
        self.agedge_data = np.load(agedge_file, allow_pickle=True)

        self.abmax_len = abmax_len
        self.agmax_len = agmax_len

        self.keys = list(self.absequence_data.keys())
        datasets = [self.absequence_data, self.abpt_data, self.abedge_data, self.agsequence_data, self.agpt_data, self.agedge_data]
        if not all(set(self.keys) == set(d.keys()) for d in datasets):
            raise ValueError("Keys in sequence, data, and edge files do not match.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        ab_sequence = self._pad_sequence(self.absequence_data[key], self.abmax_len)
        ag_sequence = self._pad_sequence(self.agsequence_data[key], self.agmax_len)

        ab_pt = self._pad_pt(self.abpt_data[key], self.abmax_len)
        ag_pt = self._pad_pt(self.agpt_data[key], self.agmax_len)

        ab_edges = torch.tensor(self.abedge_data[key], dtype=torch.long)
        ag_edges = torch.tensor(self.agedge_data[key], dtype=torch.long)

        return (
            torch.tensor(ab_sequence, dtype=torch.long),
            torch.tensor(ab_pt, dtype=torch.long),
            ab_edges,
            torch.tensor(ag_sequence, dtype=torch.long),
            torch.tensor(ag_pt, dtype=torch.long),
            ag_edges
        )

    def _pad_sequence(self, sequence, max_len):
        if sequence.shape[1] < max_len:
            sequence = np.pad(sequence, ((0, 0), (0, max_len - sequence.shape[1])), constant_values=-1)
        sequence = sequence[:, :max_len]
        if sequence.shape[0] < 100:
            sequence = np.pad(sequence, ((0, 100 - sequence.shape[0]), (0, 0)), constant_values=-1)
        return sequence[:100, :]

    def _pad_pt(self, pt, max_len):
        pt = list(pt) + [-1] * max_len
        return pt[:max_len]

