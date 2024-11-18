import torch
import csv

class PreprocessMSAData:
    def __init__(self, a3m_file1, a3m_file2, tgt_file1, tgt_file2, max_len=512, output_file="preprocessed_data.csv"):
        self.a3m_file1 = a3m_file1
        self.a3m_file2 = a3m_file2
        self.tgt_file1 = tgt_file1
        self.tgt_file2 = tgt_file2
        self.max_len = max_len
        self.output_file = output_file

        # Define the vocabulary (amino acids + padding and gap tokens)
        self.VOCAB = ["PAD", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.VOCAB)}

        # Preprocess the data
        self._preprocess_data()

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
                    return tgt_padded

    def _tokenize_and_pad(self, sequences):
        """Tokenize and pad each sequence to the maximum length."""
        tokenized_sequences = []
        for seq in sequences:
            tokenized_seq = [self.token_to_idx.get(res, self.token_to_idx['PAD']) for res in seq]
            padded_seq = (tokenized_seq + [self.token_to_idx['PAD']] * self.max_len)[:self.max_len]
            tokenized_sequences.append(padded_seq)
        return tokenized_sequences

    def _count_sequences(self):
        """Count the number of sequence groups in the first a3m file."""
        count = 0
        with open(self.a3m_file1, 'r') as f:
            for line in f:
                if line.startswith('<'):
                    count += 1
        return count

    def _preprocess_data(self):
        """Preprocess the data and save as CSV."""
        num_samples = self._count_sequences()

        # Open a CSV file for writing preprocessed data
        with open(self.output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header for the CSV file
            writer.writerow(["seq1", "seq2", "target1", "target2"])

            for idx in range(num_samples):
                # Load sequences
                print(f'writting data No.{idx}',idx)
                sequences1 = self._load_sequences(self.a3m_file1, idx)
                sequences2 = self._load_sequences(self.a3m_file2, idx)

                # Tokenize and pad
                tokenized_seq1 = self._tokenize_and_pad(sequences1)
                tokenized_seq2 = self._tokenize_and_pad(sequences2)

                # Load target sequences
                tgt1 = self._load_target(self.tgt_file1, idx)
                tgt2 = self._load_target(self.tgt_file2, idx)

                # Write data to the CSV file
                writer.writerow([tokenized_seq1, tokenized_seq2, tgt1, tgt2])

        print(f"Preprocessing complete! Data saved to {self.output_file}")


# Example usage
preprocessor = PreprocessMSAData(
    a3m_file1="chain_1.ds",
    a3m_file2="chain_2.ds",
    tgt_file1="tgt1.txt",
    tgt_file2="tgt2.txt",
    max_len=1200,
    output_file="preprocessed_data_1200.csv"
)

