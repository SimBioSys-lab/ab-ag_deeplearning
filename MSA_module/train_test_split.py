import os
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio import pairwise2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Step 1: Extract Combined Sequences by PDB ID
def extract_combined_sequences_by_pdb(chain_list_file, pdb_dir):
    """
    Combines sequences for chains from the same PDB ID.
    """
    sequences = {}
    parser = PDBParser(QUIET=True)

    with open(chain_list_file, 'r') as f:
        for line in f:
            chain_id = line.strip()
            pdb_id = chain_id[:4]
            chain = chain_id[5]
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")

            # Check if the PDB file exists
            if not os.path.isfile(pdb_file):
                print(f"PDB file {pdb_file} not found. Skipping...")
                continue

            # Parse the structure and extract the sequence for the specified chain
            structure = parser.get_structure(pdb_id, pdb_file)
            chain_seq = ""
            for model in structure:
                if chain in model:
                    seq = seq1(''.join(residue.resname for residue in model[chain].get_residues()))
                    chain_seq += seq

            # Combine chains for the same PDB ID
            if pdb_id not in sequences:
                sequences[pdb_id] = chain_seq
            else:
                sequences[pdb_id] += chain_seq

    return sequences

# Step 2: Calculate Sequence Similarity
def sequence_similarity(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    max_score = max(align.score for align in alignments)
    similarity = max_score / min(len(seq1), len(seq2))  # Normalize score
    return similarity

def compute_similarity_matrix(sequences):
    keys = list(sequences.keys())
    n = len(keys)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = sequence_similarity(sequences[keys[i]], sequences[keys[j]])
            similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    return similarity_matrix, keys

# Step 3: Cluster Based on Similarity
def cluster_sequences(similarity_matrix, threshold=0.7):
    Z = linkage(similarity_matrix, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')
    return clusters

# Step 4: Split Clusters into Training and Test Sets
def split_train_test(clusters, keys, train_ratio=0.8):
    unique_clusters = np.unique(clusters)
    train_set, test_set = [], []

    for cluster_id in unique_clusters:
        members = [keys[i] for i in range(len(keys)) if clusters[i] == cluster_id]
        if len(train_set) < train_ratio * len(keys):
            train_set.extend(members)
        else:
            test_set.extend(members)

    return train_set, test_set

# Export only PDB IDs to files
def export_to_file(filename, dataset):
    pdb_ids = set(dataset)  # To avoid duplicate PDB IDs in output
    with open(filename, 'w') as f:
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")
    print(f"Data saved to {filename}")

# Main function to execute the workflow
def main(chain_list_file, pdb_dir, train_file, test_file):
    print("Extracting combined sequences by PDB ID...", flush=True)
    sequences = extract_combined_sequences_by_pdb(chain_list_file, pdb_dir)

    print("Computing similarity matrix...", flush=True)
    similarity_matrix, keys = compute_similarity_matrix(sequences)

    print("Clustering sequences based on similarity...", flush=True)
    clusters = cluster_sequences(similarity_matrix)

    print("Splitting clusters into training and test sets...", flush=True)
    train_set, test_set = split_train_test(clusters, keys)

    print("Training set size:", len(train_set))
    print("Test set size:", len(test_set))

    # Export only PDB IDs to files
    export_to_file(train_file, train_set)
    export_to_file(test_file, test_set)

# Example usage:
chain_list_file = "/work/SimBioSys/Xing/data_collection/pdbs/ab_chains_list.txt"  # Replace with the actual file path containing chain list
pdb_dir = "/work/SimBioSys/Xing/data_collection/pdbs"  # Replace with the directory containing your PDB files
train_file = "train_set.txt"
test_file = "test_set.txt"
main(chain_list_file, pdb_dir, train_file, test_file)

