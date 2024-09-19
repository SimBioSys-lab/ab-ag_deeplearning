####A3M files from different sequences should be seperated by lines starting with "<"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class A3MDataset(Dataset):
    def __init__(self, a3m_file, max_len=512):
        self.a3m_file = a3m_file
        self.sequence_indices = self.get_sequence_indices()
        self.max_len = max_len

    def get_sequence_indices(self):
        indices = []
        with open(self.a3m_file, 'r') as f:
            pos = f.tell()
            line = f.readline()
            while line:
                if line.startswith("<"):  # Marks the start of a sequence block
                    indices.append(pos)
                pos = f.tell()
                line = f.readline()
        return indices

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        start = self.sequence_indices[idx]
        end = self.sequence_indices[idx + 1] if idx + 1 < len(self.sequence_indices) else None
        with open(self.a3m_file, 'r') as f:
            f.seek(start)
            sequences = []
            line = f.readline()
            while line and (not end or f.tell() <= end):
                # Skip headers and sequence block separators
                if not line.startswith(">") and not line.startswith("<"):
                    sequences.append(line.strip())
                line = f.readline()
        return self.tokenize(sequences)

    def tokenize(self, sequences):
        # Define the vocabulary (amino acids + padding and gap tokens)
        VOCAB = ["PAD","A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
        token_to_idx = {token: idx for idx, token in enumerate(VOCAB)}

        tokenized_sequences = []
        for sequence in sequences:
            # Ensure that both upper and lower case amino acids are handled
            tokenized_seq = [token_to_idx.get(res) for res in sequence.strip() if res in VOCAB]
            tokenized_sequences.append(tokenized_seq)

        # Pad sequences to the same length (max_len) within the dataset
        padded_sequences = [seq + [token_to_idx['PAD']] * (self.max_len - len(seq)) for seq in tokenized_sequences]
        return torch.tensor(padded_sequences, dtype=torch.long)
