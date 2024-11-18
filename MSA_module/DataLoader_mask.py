import pandas as pd
import torch
from torch.utils.data import Dataset

class PreprocessedMSADataset(Dataset):
    def __init__(self, csv_file, max_len=512):
        print("Initializing preprocessed dataset...")
        self.csv_file = csv_file
        self.max_len = max_len

        # Read the preprocessed CSV into a pandas DataFrame
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        """Return the number of preprocessed samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetch tokenized and padded sequences and targets from the CSV."""
        row = self.data.iloc[idx]  # Access the row using pandas

        masked_seq1 = torch.tensor(eval(row[0]), dtype=torch.long)
        tgt1 = torch.tensor(eval(row[2]), dtype=torch.long)
        mask_pos1 = int(row[4])

        return masked_seq1, tgt1,  mask_pos1

