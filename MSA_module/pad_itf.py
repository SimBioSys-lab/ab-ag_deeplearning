import numpy as np

# Configuration
Threshold = 1600  # Set your desired threshold
output_file = f"ihd_test_esminterfaces10_{Threshold}.npz"  # Output file for padded interfaces

# Load the combined_test_interfaces.npz file
interfaces = np.load("ihd_test_esminterfaces10.npz", allow_pickle=True)

# Dictionary to store processed interfaces
padded_interfaces = {}

for key in interfaces.keys():
    interface = interfaces[key]  # Retrieve the interface array
    length = len(interface)  # Get the length of the array

    # Check if the interface length exceeds the threshold
    if length > Threshold:
        print(f"Interface for key {key} exceeds threshold ({length} > {Threshold}). Dropping...")
        continue

    # Pad the interface to Threshold with -1
    padded_interface = np.full(Threshold, -1, dtype=interface.dtype)
    padded_interface[:length] = interface  # Copy the original interface into the padded array

    # Store the padded interface
    padded_interfaces[key] = padded_interface

# Save the processed interfaces to a new NPZ file
np.savez_compressed(output_file, **padded_interfaces)
print(f"Padded interfaces saved to {output_file}.")

