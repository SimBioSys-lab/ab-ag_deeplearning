import torch
import csv

class PreprocessMSAData:
    def __init__(self, a3m_file1, a3m_file2, max_len=512, output_file="preprocessed_data.csv"):
        self.a3m_file1 = a3m_file1
        self.a3m_file2 = a3m_file2
        self.max_len = max_len
        self.output_file = output_file

        # Define the vocabulary (amino acids + padding and gap tokens)
        self.VOCAB = ["PAD", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.VOCAB)}
        self.vocab_size = len(self.VOCAB)

        # Preprocess the data
        self._preprocess_data()

    def _load_sequences(self, a3m_file, idx):
        """Load sequences for a specific block from the a3m file based on the index."""
        current_idx = -1
        sequence_block = []
        with open(a3m_file, 'r') as f:
            for line in f:
                if line.startswith('<'):
                    current_idx += 1
                    if current_idx > idx:
                        break
                if current_idx == idx and not line.startswith('<'):
                    sequence_block.append(line.strip())
        return sequence_block

    def _tokenize_and_pad(self, sequences):
        """Tokenize and pad each sequence to the maximum length."""
        tokenized_sequences = []
        for seq in sequences:
            tokenized_seq = [self.token_to_idx.get(res, self.token_to_idx['PAD']) for res in seq]
            padded_seq = (tokenized_seq + [self.token_to_idx['PAD']] * self.max_len)[:self.max_len]
            tokenized_sequences.append(padded_seq)
        return torch.tensor(tokenized_sequences, dtype=torch.long)

    def _count_sequences(self):
        """Count the number of sequence groups in the first a3m file."""
        count = 0
        with open(self.a3m_file1, 'r') as f:
            for line in f:
                if line.startswith('<'):
                    count += 1
        return count

    def _preprocess_data(self):
        """Preprocess the data and save it as CSV."""
        num_samples = self._count_sequences()

        # Open a CSV file for writing preprocessed data
        with open(self.output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header for the CSV file
            writer.writerow(["masked_seq1", "masked_seq2", "tgt1", "tgt2", "mask_pos1", "mask_pos2"])

            for idx in range(num_samples):
                # Load sequences
                sequences1 = self._load_sequences(self.a3m_file1, idx)
                sequences2 = self._load_sequences(self.a3m_file2, idx)

                # Tokenize and pad
                tokenized_seq1 = self._tokenize_and_pad(sequences1)
                tokenized_seq2 = self._tokenize_and_pad(sequences2)

                # Get original sequences for masking
                original_seq1 = tokenized_seq1[0]
                original_seq2 = tokenized_seq2[0]

                # Mask each residue in the first row (original sequence) individually
                for mask_pos1 in range(len(original_seq1)):
                    if original_seq1[mask_pos1] != self.token_to_idx["PAD"] and original_seq1[mask_pos1] != self.token_to_idx["-"]:
                        masked_seq1 = tokenized_seq1.clone()
                        masked_seq1[0, mask_pos1] = self.token_to_idx["PAD"]  # Mask this residue
                        tgt1 = original_seq1[mask_pos1]  # The original target

                        for mask_pos2 in range(len(original_seq2)):
                            if original_seq2[mask_pos2] != self.token_to_idx["PAD"] and original_seq2[mask_pos2] != self.token_to_idx["-"]:
                                masked_seq2 = tokenized_seq2.clone()
                                masked_seq2[0, mask_pos2] = self.token_to_idx["PAD"]  # Mask this residue
                                tgt2 = original_seq2[mask_pos2]  # The original target

                                # Write the masked sequences and targets to the CSV file
                                writer.writerow([
                                    masked_seq1.tolist(), 
                                    masked_seq2.tolist(), 
                                    tgt1.item(), 
                                    tgt2.item(), 
                                    mask_pos1, 
                                    mask_pos2
                                ])

        print(f"Preprocessing complete! Data saved to {self.output_file}")

# Example usage
preprocessor = PreprocessMSAData(
    a3m_file1="chain_1.ds",
    a3m_file2="chain_2.ds",
    max_len=1200,
    output_file="preprocessed_data_mask_1200.csv"
)

