import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from Bio import SeqIO

def downsample_msa(msa_file, method="clustering", num_sequences=100):
    """Downsamples an MSA to a specified number of sequences.

    Args:
        msa_file (str): Path to the MSA file in FASTA format.
        method (str, optional): Downsampling method. Can be "clustering" or "random".
            Default is "clustering".
        num_sequences (int, optional): Number of sequences to retain in the downsampled MSA.
            Default is 100.

    Returns:
        list: List of downsampled sequences.
    """

    # Load MSA sequences
    sequences = []
    with open(msa_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences.append(str(record.seq))

    if method == "clustering":
        # Calculate pairwise distances between sequences
        distances = np.zeros((len(sequences), len(sequences)))
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                distances[i, j] = 1 - pairwise_alignment(sequences[i], sequences[j])
                distances[j, i] = distances[i, j]

        # Perform hierarchical clustering
        Z = linkage(distances, method="average")

        # Assign clusters to sequences
        clusters = fcluster(Z, t=num_sequences, criterion="maxclust")

        # Select representative sequences from each cluster
        downsampled_sequences = []
        for cluster_id in range(1, num_sequences + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                downsampled_sequences.append(sequences[cluster_indices[0]])

    elif method == "random":
        # Randomly select sequences
        downsampled_sequences = np.random.choice(sequences, size=num_sequences, replace=False).tolist()

    else:
        raise ValueError("Invalid downsampling method. Must be 'clustering' or 'random'.")

    return downsampled_sequences

def pairwise_alignment(seq1, seq2):
    """Calculates the pairwise sequence identity between two sequences.

    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.

    Returns:
        float: Pairwise sequence identity.
    """

    # You can use a suitable alignment algorithm like Needleman-Wunsch or Smith-Waterman
    # Here's a simplified example using Needleman-Wunsch:
    from Bio import pairwise2

    alignments = pairwise2.align.globalxx(seq1, seq2)
    if alignments:
        best_alignment = alignments[0]
        identity = best_alignment[3] / len(best_alignment[0])
        return identity
    else:
        return 0.0

# Example usage
msa_file = "your_msa.fasta"
downsampled_sequences = downsample_msa(msa_file, method="clustering", num_sequences=50)

# Print the downsampled sequences
for sequence in downsampled_sequences:
    print(sequence)
