import torch
from torch.utils.data import Dataset

class MSADataset(Dataset):
    def __init__(self, a3m_file1, a3m_file2, tgt_file1, tgt_file2, max_len=512):
        print("Initializing dataset...")
        # Store file paths instead of loading data into memory
        self.a3m_file1 = a3m_file1
        self.a3m_file2 = a3m_file2
        self.tgt_file1 = tgt_file1
        self.tgt_file2 = tgt_file2
        self.max_len = max_len
       
        # Define the vocabulary (amino acids + padding and gap tokens)
        self.VOCAB = ["PAD", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.VOCAB)}

        # Calculate the number of sequence groups (blocks) once during initialization
        self.num_samples = self._count_sequences()

    def _count_sequences(self):
        """Count the number of sequence groups in the first a3m file."""
        count = 0
        with open(self.a3m_file1, 'r') as f:
            for line in f:
                if line.startswith('<'):
                    count += 1
        return count

    def _load_sequences(self, a3m_file, idx):
        """Load sequences for a specific group from the a3m file based on the index."""
        sequences = []
        current_idx = -1
        current_sequences = []

        with open(a3m_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('<'):
                    current_idx += 1
                    if current_idx > idx:
                        break
                    if current_idx == idx:
                        current_sequences = []
                elif current_idx == idx:
                    current_sequences.append(line)
                   
        return current_sequences

    def _load_target(self, tgt_file, idx):
        """Load target sequence for a specific group from the target file based on the index."""
        with open(tgt_file, 'r') as f:
            for current_idx, line in enumerate(f):
                if current_idx == idx:
                    tgt = [int(x) for x in line.split()]
                    tgt_padded = (tgt + [-1] * self.max_len)[:self.max_len]
                    return torch.tensor(tgt_padded, dtype=torch.int)

    def _tokenize_and_pad(self, sequences):
        """Tokenize and pad each sequence to the maximum length."""
        tokenized_sequences = []
        for seq in sequences:
            tokenized_seq = [self.token_to_idx.get(res, self.token_to_idx['PAD']) for res in seq]
            padded_seq = (tokenized_seq + [self.token_to_idx['PAD']] * self.max_len)[:self.max_len]
            tokenized_sequences.append(padded_seq)

        return torch.tensor(tokenized_sequences, dtype=torch.long)

    def __len__(self):
        """Return the number of sequence groups (blocks) in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        print(f"Fetching item {idx} from the dataset...")
        # Load sequences on-demand
        sequences1 = self._load_sequences(self.a3m_file1, idx)
        sequences2 = self._load_sequences(self.a3m_file2, idx)

        # Tokenize and pad sequences
        tokenized_seq1 = self._tokenize_and_pad(sequences1)
        tokenized_seq2 = self._tokenize_and_pad(sequences2)

        # Load target sequences on-demand
        padded_tgt1 = self._load_target(self.tgt_file1, idx)
        padded_tgt2 = self._load_target(self.tgt_file2, idx)

        return tokenized_seq1, tokenized_seq2, padded_tgt1, padded_tgt2
