import pyrosetta
import numpy as np
from pyrosetta import pose_from_sequence
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.id import AtomID

# ---------------------------
# 1. Initialize PyRosetta
# ---------------------------
pyrosetta.init()

# ---------------------------
# 2. Define Sequences for Each Chain and Create Individual Poses
# ---------------------------
# Example sequences for two chains.
chain1_seq = "ACDEFGHIK"      # Chain A
chain2_seq = "LMNPQRSTVWY"    # Chain B

# Create poses for each chain.
pose_chain1 = pose_from_sequence(chain1_seq)
pose_chain2 = pose_from_sequence(chain2_seq)

# ---------------------------
# 3. Combine the Chains into a Single Multi-Chain Pose
# ---------------------------
# Append the second chain as a new chain to the first pose.
pose_chain1.append_pose_by_jump(pose_chain2, new_chain=True)
multi_chain_pose = pose_chain1

# Verify chain assignment (global numbering and chain IDs)
print("Residue\tChain")
for i in range(1, multi_chain_pose.total_residue() + 1):
    print(f"{i}\t{multi_chain_pose.pdb_info().chain(i)}")

# ---------------------------
# 4. Map Chain Letters to Global Starting Residue Numbers
# ---------------------------
# When building a multi-chain pose, residues from different chains have global numbering.
# We'll build a simple mapping from chain letter (as stored in pdb_info) to the first global residue index of that chain.
chain_start_map = {}
for i in range(1, multi_chain_pose.total_residue() + 1):
    chain_letter = multi_chain_pose.pdb_info().chain(i)
    if chain_letter not in chain_start_map:
        chain_start_map[chain_letter] = i

# For example, if chain 'A' starts at residue 1 and chain 'B' starts at residue 10.
print("Chain start mapping:", chain_start_map)

# ---------------------------
# 5. Define a Sample Contact Map and Add Constraints
# ---------------------------
# Suppose we have two types of contacts:
#   a) Intra-chain contact in chain A: between residue 2 and residue 7 (local indices)
#   b) Inter-chain contact: between chain A residue 3 and chain B residue 4
#
# Contacts are defined using chain letter and residue number in that chain.
contacts = [
    # Format: (chain1_letter, res1_local, chain2_letter, res2_local)
    ("A", 2, "A", 7),   # Intra-chain contact in chain A
    ("A", 3, "B", 4)    # Inter-chain contact: chain A to chain B
]

# Loop over contacts and add a constraint for each.
for contact in contacts:
    chain1_letter, res1_local, chain2_letter, res2_local = contact

    # Convert chain-local indices to global residue numbers.
    global_res1 = chain_start_map[chain1_letter] + res1_local - 1
    global_res2 = chain_start_map[chain2_letter] + res2_local - 1

    # Define a harmonic restraint for the CA-CA distance.
    target_distance = 8.0  # target distance (in Å)
    std_dev = 2.0          # standard deviation for the restraint
    func = HarmonicFunc(target_distance, std_dev)

    # Get the CA atom indices from the residues.
    atom1 = AtomID(multi_chain_pose.residue(global_res1).atom_index("CA"), global_res1)
    atom2 = AtomID(multi_chain_pose.residue(global_res2).atom_index("CA"), global_res2)

    # Create the constraint and add it to the pose.
    constraint = AtomPairConstraint(atom1, atom2, func)
    multi_chain_pose.add_constraint(constraint)

# ---------------------------
# 6. Set Up the Scoring Function and Relax Protocol
# ---------------------------
# Use a score function that includes constraint terms.
scorefxn = pyrosetta.create_score_function("ref2015_cst")

# Use FastRelax with constraints.
from pyrosetta.rosetta.protocols.relax import FastRelax
relax = FastRelax()
relax.set_scorefxn(scorefxn)

# Apply the relax protocol to refine the structure.
relax.apply(multi_chain_pose)

# ---------------------------
# 7. Save the Final Multi-Chain Structure
# ---------------------------
multi_chain_pose.dump_pdb("final_multi_chain_structure.pdb")
print("Final multi-chain structure written to 'final_multi_chain_structure.pdb'.")

