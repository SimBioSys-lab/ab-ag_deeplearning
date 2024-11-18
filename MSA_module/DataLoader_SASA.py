import pandas as pd
import torch
from torch.utils.data import Dataset

class PreprocessedMSADataset(Dataset):
    def __init__(self, csv_file, max_len=1200):
        print("Initializing preprocessed dataset...", flush=True)
        self.csv_file = csv_file
        self.max_len = max_len

        # Read the CSV into a pandas DataFrame
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        """Return the number of preprocessed samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetch tokenized and padded sequences and targets from the CSV."""
        row = self.data.iloc[idx]  # Access the row using pandas iloc for positional indexing

        # Assuming the columns are in the same order as the previous example
        tokenized_seq1 = torch.tensor(eval(row.iloc[0]), dtype=torch.long)
        tokenized_seq2 = torch.tensor(eval(row.iloc[1]), dtype=torch.long)
        tgt1 = torch.tensor(eval(row.iloc[2]), dtype=torch.long)
        tgt2 = torch.tensor(eval(row.iloc[3]), dtype=torch.long)

        return tokenized_seq1, tokenized_seq2, tgt1, tgt2

