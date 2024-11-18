import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import distance_matrix

def extract_adjacency_matrix(pdb_file, distance_threshold=5.0):
    """
    Extracts an adjacency matrix from a PDB file based on a distance threshold.
    
    Args:
        pdb_file (str): Path to the PDB file.
        distance_threshold (float): The distance threshold for considering atoms as adjacent.
    
    Returns:
        np.ndarray: The adjacency matrix.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Extract all atom coordinates (CA atoms for simplicity) and atom IDs
    atom_coords = []
    atom_ids = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:  # Only use alpha carbons (CA) for adjacency
                    atom = residue['CA']
                    atom_coords.append(atom.coord)
                    atom_ids.append((chain.id, residue.id[1]))  # (chain_id, residue_id)
    
    # Convert atom coordinates to a NumPy array
    atom_coords = np.array(atom_coords)
    
    # Calculate the distance matrix
    dist_matrix = distance_matrix(atom_coords, atom_coords)
    
    # Create the adjacency matrix based on the distance threshold
    adj_matrix = (dist_matrix < distance_threshold).astype(int)
    
    # Remove self-connections (diagonal should be zero)
    np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix, atom_ids

# Example usage
pdb_file = "your_protein.pdb"
adj_matrix, atom_ids = extract_adjacency_matrix(pdb_file, distance_threshold=5.0)

print("Adjacency Matrix:\n", adj_matrix)
print("Atom IDs (chain, residue):\n", atom_ids)

