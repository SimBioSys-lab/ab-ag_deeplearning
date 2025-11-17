import numpy as np

def combine_npz(npz_file1, npz_file2, output_file, conflict_strategy='check'):
    """
    Combine two .npz files into one.

    Args:
        npz_file1 (str): Path to the first .npz file.
        npz_file2 (str): Path to the second .npz file.
        output_file (str): Path for the output combined .npz file.
        conflict_strategy (str): Strategy to handle keys present in both files. Options:
            - 'check': Check if the arrays are equal (using allclose); if not, warn and use npz_file2's value.
            - 'overwrite': Always use the value from npz_file2.
            - 'skip': Keep the value from npz_file1 and ignore npz_file2's value.
    """
    # Load the NPZ files.
    data1 = np.load(npz_file1, allow_pickle=True)
    data2 = np.load(npz_file2, allow_pickle=True)

    # Combine the dictionaries.
    combined = {}

    # Add keys from the first file.
    for key in data1.keys():
        combined[key] = data1[key]

    # Iterate over keys in the second file.
    for key in data2.keys():
        if key in combined:
            if conflict_strategy == 'check':
                # For numeric arrays, compare using np.allclose.
                val1 = combined[key]
                val2 = data2[key]
                if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                    if not np.allclose(val1, val2, atol=1e-8, rtol=1e-5):
                        print(f"Warning: Key '{key}' differs between files. Overwriting with value from npz_file2.")
                        combined[key] = val2
                else:
                    if val1 != val2:
                        print(f"Warning: Key '{key}' differs between files. Overwriting with value from npz_file2.")
                        combined[key] = val2
            elif conflict_strategy == 'overwrite':
                combined[key] = data2[key]
            elif conflict_strategy == 'skip':
                # Skip adding from file2 if key already exists.
                continue
            else:
                raise ValueError("conflict_strategy must be one of ['check', 'overwrite', 'skip']")
        else:
            combined[key] = data2[key]

    # Save the combined dictionary to a new .npz file.
    np.savez(output_file, **combined)
    print(f"Combined NPZ file saved to: {output_file}")

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Set the file paths for your input NPZ files and the output file.
    npz_file1 = "para_train_esmss.npz"
    npz_file2 = "para_val_esmss.npz"
    output_file = "para_tv_esmss.npz"

    # Combine the NPZ files.
    combine_npz(npz_file1, npz_file2, output_file, conflict_strategy='skip')

