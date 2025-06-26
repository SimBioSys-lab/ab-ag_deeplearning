import numpy as np

# Load the NPZ files
padded_npz = np.load('padded_sequences_train_filtered_2400.npz')
antibody_npz = np.load('padded_train_sasa_2400.npz')

# Get the keys from the padded sequences file
padded_keys = set(padded_npz.files)

# Create a dictionary to hold the aligned antibody data
aligned_antibody = {key: antibody_npz[key] for key in antibody_npz.files if key in padded_keys}

# Save the aligned antibody data to a new NPZ file
np.savez('padded_train_sasa_filtered_2400.npz', **aligned_antibody)

print(f"Aligned {len(aligned_antibody)} keys from antibody data with the padded sequences.")

