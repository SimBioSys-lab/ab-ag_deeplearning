from Bio import PDB
import os
import sys
def pdb_to_fasta(pdb_file):
    # Extract the base name of the PDB file (without the .pdb extension)
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    
    # Parse the PDB file
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb_name, pdb_file)
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            seq = ""
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    seq += PDB.Polypeptide.three_to_one(residue.resname)
            
            # Save the sequence to a FASTA file, including the PDB name and chain ID in the filename
            fasta_filename = f"{pdb_name}_chain_{chain_id}.fasta"
            with open(fasta_filename, "w") as f:
                f.write(f">{pdb_name}_Chain_{chain_id}\n{seq}\n")
            print(f"FASTA file written: {fasta_filename}")

# Call the function with your PDB file
pdb_name = sys.argv[1]
pdb_to_fasta(pdb_name)

