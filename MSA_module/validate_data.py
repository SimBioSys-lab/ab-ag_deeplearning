import numpy as np

# Ensure full array printing (no abbreviation)
np.set_printoptions(threshold=np.inf)

# Load the NPZ files
seq = np.load('padded_sequences_test_3000.npz')
itf = np.load('padded_interfaces_test_3000.npz')
edge = np.load('padded_edges_test_3000.npz')

for key in seq.keys():
    if key in itf.keys():
        # Create masks for valid values
        seq_mask = (seq[key][0] != 0) & (seq[key][0] != 22)  # Exclude padding and EOC
        itf_mask = (itf[key] != -1)  # Exclude padding

        # Check if the masks match
        if not np.array_equal(seq_mask, itf_mask):
            print(f"Masks do not match for {key}.")
            mismatch_indices = np.where(seq_mask != itf_mask)[0]
            print(f"Mismatch indices for {key}: {mismatch_indices}")

# Print details for the specific key "3ztn"
key = '3ztn'
if key in seq.keys():
    print(f"\nDetails for key '{key}':\n")
    print("Sequence:")
    print(seq[key][0].tolist())
    print("Indices of EOC (value 22):")
    print(np.where(seq[key][0] == 22))
    print("\nInterfaces:")
    print(itf[key])
    print("\nEdges:")
    for edge_info in edge[key]:
        print(edge_info)
else:
    print(f"Key '{key}' not found in the sequence data.")

