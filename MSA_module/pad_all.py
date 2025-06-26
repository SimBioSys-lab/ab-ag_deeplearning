import numpy as np

# Configuration
Threshold = 1600
output_prefix = f"cleaned_para_tv"

# Load input files
train_seqs = np.load("cleaned_para_tv_sequences.npz", allow_pickle=True)
interfaces = np.load("cleaned_para_tv_interfaces.npz", allow_pickle=True)
edges = np.load("cleaned_para_tv_edges.npz", allow_pickle=True)
#test_seqs = np.load("cleaned_para_test_sequences_1600.npz", allow_pickle=True)

# Helper: split sequence into chains based on [0] separator
def split_chains(seq_row):
    parts = []
    current = []
    for token in seq_row:
        if token == 22:
            if current:
                parts.append(np.array(current, dtype=int))
                current = []
        else:
            current.append(token)
    if current:
        parts.append(np.array(current, dtype=int))
    return parts

# Compute sequence identity between two chains
def sequence_identity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    return np.sum(seq1[:min_len] == seq2[:min_len]) / min_len

# Return True if any chain from train has ≥30% identity to any test chain
def has_high_similarity(train_seq, test_seq_dict, threshold=0.9):
    train_chains = split_chains(train_seq[0])
    for test_seq in test_seq_dict.values():
        test_chains = split_chains(test_seq[0])
        for tr in train_chains:
            for ts in test_chains:
                if sequence_identity(tr, ts) >= threshold:
                    return True
    return False

# Output dictionaries
padded_sequences = {}
padded_interfaces = {}
filtered_edges = {}
valid_keys = []

# Main loop
for key in train_seqs.keys():
    if key not in interfaces:
        print(f"Skipping {key}: no interface.")
        continue

    seq = train_seqs[key]
    interface = interfaces[key]
    seq_rows, seq_cols = seq.shape
    itf_len = len(interface)

    if seq_cols > Threshold or itf_len > Threshold:
        print(f"Dropping {key}: seq={seq_cols}, itf={itf_len} > {Threshold}")
        continue

#    if has_high_similarity(seq, test_seqs):
#        print(f"Dropping {key}: ≥30% identity to test set")
#        continue

    # Pad sequence
    padded_seq = np.zeros((64, Threshold), dtype=seq.dtype)
#    padded_seq[:seq_rows, :seq_cols] = seq
    padded_seq[:64, :seq_cols] = seq[:64,:seq_cols]
    padded_sequences[key] = padded_seq

    # Pad interface
    padded_itf = np.full(Threshold, -1, dtype=interface.dtype)
    padded_itf[:itf_len] = interface
    padded_interfaces[key] = padded_itf

    valid_keys.append(key)

# Collect edges for valid entries
for key in valid_keys:
    if key in edges:
        filtered_edges[key] = edges[key]

# Save results
np.savez_compressed(f"{output_prefix}_sequences_{Threshold}.npz", **padded_sequences)
np.savez_compressed(f"{output_prefix}_interfaces_{Threshold}.npz", **padded_interfaces)
np.savez_compressed(f"{output_prefix}_edges_{Threshold}.npz", **filtered_edges)

print("Saved filtered sequences with chain-wise similarity <30% to test set.")

