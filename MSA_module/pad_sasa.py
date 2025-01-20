import numpy as np

# Configuration
Threshold = 3000  # Set your desired threshold
output_file = f"padded_test_sasa_{Threshold}.npz"  # Output file for padded SASA data

# Load the SASA data file
sasa_data = np.load("combined_test_sasa.npz", allow_pickle=True)

# Dictionary to store processed SASA data
padded_sasa_data = {}

for key in sasa_data.keys():
    sasa_array = sasa_data[key]  # Retrieve the SASA array
    length = len(sasa_array)  # Get the length of the array

    # Check if the SASA array length exceeds the threshold
    if length > Threshold:
        print(f"SASA data for key {key} exceeds threshold ({length} > {Threshold}). Dropping...")
        continue

    # Pad the SASA array to Threshold with -1
    padded_sasa = np.full(Threshold, -1, dtype=sasa_array.dtype)
    padded_sasa[:length] = sasa_array  # Copy the original SASA array into the padded array

    # Store the padded SASA array
    padded_sasa_data[key] = padded_sasa

# Save the processed SASA data to a new NPZ file
np.savez_compressed(output_file, **padded_sasa_data)
print(f"Padded SASA data saved to {output_file}.")

