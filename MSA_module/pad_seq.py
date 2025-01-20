import numpy as np

# Configuration
Threshold = 3000  # Set your desired threshold for sequence length
output_file = f"test_train_sequences_{Threshold}.npz"  # Output file for padded sequences

# Load the combined_train_sequences.npz file
sequences = np.load("combined_train_sequences.npz", allow_pickle=True)

# Dictionary to store processed sequences
padded_sequences = {}

for key in sequences.keys():
    seq = sequences[key]  # Retrieve the sequence matrix
    rows, cols = seq.shape  # Get the dimensions of the matrix

    # Check if the sequence length exceeds the threshold
    if cols > Threshold:
        print(f"Sequence for key {key} exceeds threshold ({cols} > {Threshold}). Dropping...")
        continue

    # Pad the sequence to 100 * Threshold with 0
    padded_seq = np.zeros((100, Threshold), dtype=seq.dtype)
    padded_seq[:rows, :cols] = seq  # Copy the original sequence matrix into the padded matrix

    # Store the padded sequence
    padded_sequences[key] = padded_seq

# Save the processed sequences to a new NPZ file
np.savez_compressed(output_file, **padded_sequences)
print(f"Padded sequences saved to {output_file}.")

