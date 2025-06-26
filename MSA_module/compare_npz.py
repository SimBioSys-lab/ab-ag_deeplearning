import numpy as np

def compare_npz_files(npz_file1, npz_file2, atol=1e-8, rtol=1e-5):
    # Load NPZ files
    data1 = np.load(npz_file1, allow_pickle=True)
    data2 = np.load(npz_file2, allow_pickle=True)

    # Get set of keys common to both files
    common_keys = set(data1.keys()) & set(data2.keys())
    print("Common keys:", common_keys)

    all_equal = True

    for key in common_keys:
        value1 = data1[key][0]
        value2 = data2[key][0]
        
        # If they are numpy arrays, use allclose for numerical comparison.
        # Otherwise, use equality operator.
        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            equal = np.allclose(value1, value2, atol=atol, rtol=rtol)
        else:
            equal = (value1 == value2)
        
        print(f"Key: {key} -> Equal: {equal}")
        if not equal:
            all_equal = False

    if all_equal:
        print("All matching keys have the same value.")
    else:
        print("Some keys have different values.")

if __name__ == "__main__":
    # Replace these with your NPZ file paths
    npz_file1 = "cleaned_para_tv_sequences_1600.npz"
    npz_file2 = "cleaned_para_test_sequences_1600.npz"
    compare_npz_files(npz_file1, npz_file2)

