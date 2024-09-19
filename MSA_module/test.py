import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

max_len = 512
a3m_file = 'test_2.a3m'
indices = []
with open(a3m_file, 'r') as f:
    pos = f.tell()
    line = f.readline()
    while line:
        if line.startswith("<"):  # Marks the start of a sequence block
            indices.append(pos)
        pos = f.tell()
        line = f.readline()
idx = 0
start = indices[idx]
print("start",start)
end = indices[idx + 1] if idx + 1 < len(indices) else None
print("end",end)
with open(a3m_file, 'r') as f:
    f.seek(start)
    print(line)
    sequences = []
    line = f.readline()
    while line and (not end or f.tell() <= end):
        # Skip headers and sequence block separators
        if not line.startswith(">") and not line.startswith("<"):
            sequences.append(line.strip())
        line = f.readline()
print("sequences",sequences)
# Define the vocabulary (amino acids + padding and gap tokens)
VOCAB = ["PAD","A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
token_to_idx = {token: idx for idx, token in enumerate(VOCAB)}

tokenized_sequences = []
for sequence in sequences:
    # Ensure that both upper and lower case amino acids are handled
    tokenized_seq = [token_to_idx.get(res) for res in sequence.strip() if res in VOCAB]
    tokenized_sequences.append(tokenized_seq)

# Pad sequences to the same length (max_len) within the dataset
padded_sequences = [seq + [token_to_idx['PAD']] * (max_len - len(seq)) for seq in tokenized_sequences]

print(padded_sequences)
print(len(padded_sequences))
