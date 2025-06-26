import numpy as np

def truncate_npz(input_path, output_path, max_length=2400):
    # Load the NPZ file. Using allow_pickle=True if needed.
    data = np.load(input_path, allow_pickle=True)

    # Dictionary to store truncated arrays.
    truncated_data = {}
            
    # Loop over each stored array in the NPZ file.
    for key in data.files:
        array = data[key]
        # Slice the array to keep only the first `max_length` elements along the first axis.
        truncated_array = array[:max_length,:max_length]
        truncated_data[key] = truncated_array
    np.savez(output_path, **truncated_data)
    print(f"Truncated data saved to {output_path}")
if __name__ == "__main__":
    input_npz = 'global_maps_test_aligned_3000.npz'   # Replace with your input file path
    output_npz = 'global_maps_test_aligned_2400.npz' # Replace with your desired output file path
    truncate_npz(input_npz, output_npz)
