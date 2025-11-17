import numpy as np

# Configuration
Threshold = 1600  # Set your desired threshold
output_file = f"para_tv_esmss_{Threshold}.npz"  # Output file for padded secondary structure data

# Load the secondary structure data file
secondary_structure_data = np.load("para_tv_esmss.npz", allow_pickle=True)

# Dictionary to store processed secondary structure data
padded_secondary_structure = {}

for key in secondary_structure_data.keys():
    ss_array = secondary_structure_data[key]  # Retrieve the tokenized secondary structure array
    length = len(ss_array)  # Get the length of the array

    # Check if the array length exceeds the threshold
    if length > Threshold:
        print(f"Secondary structure data for key {key} exceeds threshold ({length} > {Threshold}). Dropping...")
        continue

    # Pad the array to Threshold with -1
    padded_ss = np.full(Threshold, -1, dtype=ss_array.dtype)
    padded_ss[:length] = ss_array  # Copy the original array into the padded array

    # Store the padded secondary structure data
    padded_secondary_structure[key] = padded_ss

# Save the processed secondary structure data to a new NPZ file
np.savez_compressed(output_file, **padded_secondary_structure)
print(f"Padded secondary structure data saved to {output_file}.")

