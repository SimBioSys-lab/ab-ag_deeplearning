import numpy as np
import matplotlib.pyplot as plt

# Load .npz file where each key is a PDB ID and value is a sequence array
data = np.load("MIPE_tv_esminterfaces.npz", allow_pickle=True)

# Chain length trackers
l_lengths = []
h_lengths = []
ag_lengths = []

# Loop over each PDB entry
for pdbid in data.files:
    combined = data[pdbid]  # This is a numpy array, not a string

    # Convert to list of tokens (e.g., characters or IDs)
    sequence_tokens = combined.tolist()

    # Split by delimiter -1
    chains = []
    current_chain = []
    for token in sequence_tokens:
        if token == -1:
            if current_chain:
                chains.append(current_chain)
                current_chain = []
        else:
            current_chain.append(token)
    if current_chain:
        chains.append(current_chain)

    # Need at least L, H, and one AG chain
    if len(chains) < 3:
        print(f"Skipping {pdbid}: fewer than 3 chains found.")
        continue

    l_lengths.append(len(chains[0]))  # Lchain
    h_lengths.append(len(chains[1]))  # Hchain
    ag_lengths.extend(len(ag) for ag in chains[2:])  # all AG chains

# === Plot the distributions ===
plt.figure(figsize=(10, 6))

plt.hist(l_lengths, bins=30, alpha=0.7, label='Lchain')
plt.hist(h_lengths, bins=30, alpha=0.7, label='Hchain')
plt.hist(ag_lengths, bins=30, alpha=0.7, label='AG chains')

plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Chain Length Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

