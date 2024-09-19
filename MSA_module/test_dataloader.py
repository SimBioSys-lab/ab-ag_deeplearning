from DataLoader_att import A3MDataset 
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np  # Only if you're using NumPy for preprocessing or other array operations

a3m_file = "test_256.a3m"
dataset = A3MDataset(a3m_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(batch)  # Print the actual batch data (tokens or sequences)
    print(f"Shape of the batch: {batch.shape}")  # Check the shape of the batch

