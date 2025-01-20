import numpy as np

# File paths
edges_file = "padded_edges_3000.npz"
interfaces_file = "padded_interfaces_3000.npz"
sequences_file = "padded_sequences_3000.npz"

# Load the data from the NPZ files
edges_data = np.load(edges_file, allow_pickle=True)
interfaces_data = np.load(interfaces_file, allow_pickle=True)
sequences_data = np.load(sequences_file, allow_pickle=True)

# Find common keys among the three files
common_keys = set(edges_data.keys()) & set(interfaces_data.keys()) & set(sequences_data.keys())
print(f"Number of common keys: {len(common_keys)}")

# Filter data to retain only common keys
filtered_edges = {key: edges_data[key] for key in common_keys}
filtered_interfaces = {key: interfaces_data[key] for key in common_keys}
filtered_sequences = {key: sequences_data[key] for key in common_keys}

# Save filtered data back to new NPZ files
np.savez_compressed("filtered_edges_3000.npz", **filtered_edges)
np.savez_compressed("filtered_interfaces_3000.npz", **filtered_interfaces)
np.savez_compressed("filtered_sequences_3000.npz", **filtered_sequences)

print("Filtered NPZ files saved with common keys.")

