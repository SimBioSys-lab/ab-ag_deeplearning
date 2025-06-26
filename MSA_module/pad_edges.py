import numpy as np

# Load the NPZ file
edges_file = "global_maps_train10.npz"
edges_data = np.load(edges_file, allow_pickle=True)

# Keys to be removed
keys_to_remove = {"7x25", "7x29", "7we8", "7nda", "7k8u", "7kcr", "7cac"}

# Filter the keys to retain only the desired ones
filtered_data = {key: edges_data[key] for key in edges_data if key not in keys_to_remove}

# Save the filtered data back to a new NPZ file
output_file = "global_maps_train10_1600.npz"
np.savez_compressed(output_file, **filtered_data)

print(f"Removed keys: {keys_to_remove}")
print(f"Filtered NPZ saved as '{output_file}'")

