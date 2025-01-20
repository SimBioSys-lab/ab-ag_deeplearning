import numpy as np

# File paths
seq_file = "preprocessed_seq_ab_train_1200.npz"
edge_file = "train_edge_lists.npz"

# Load the .npz files
seq_data = np.load(seq_file)
edge_data = np.load(edge_file)

# Get the keys
seq_keys = seq_data.files
edge_keys = edge_data.files

# Print the keys for inspection
print("Keys in preprocessed_seq_ab_train_1200.npz:", seq_keys)
print("Keys in train_edge_lists.npz:", edge_keys)

# Compare the keys
if set(seq_keys) == set(edge_keys):
    print("\nThe keys in both files match!")
else:
    print("\nThe keys do not match.")
    print("Keys in seq_data but not in edge_data:", set(seq_keys) - set(edge_keys))
    print("Keys in edge_data but not in seq_data:", set(edge_keys) - set(seq_keys))

# Close the files (optional but recommended)
seq_data.close()
edge_data.close()

