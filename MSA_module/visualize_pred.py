import csv

# Filenames (adjust as needed)
csv_filename = "7a3p_10.csv"      # CSV file with header: Chain Type,True Label,Predicted Label,Probability
pdb_filename = "7a3p.pdb"       # Original PDB file

# Initialize dictionaries to store labels per chain type.
chain_labels_true = {"Lchain": [], "Hchain": [], "AGchain_0": []}
chain_labels_pred = {"Lchain": [], "Hchain": [], "AGchain_0": []}

# Read the CSV file and populate the dictionaries.
with open(csv_filename, "r") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        chain_type = row["Chain Type"].strip()  # Should be one of Lchain, Hchain, AGchain.
        if chain_type in chain_labels_true:
            chain_labels_true[chain_type].append(float(row["True Label"]))
            chain_labels_pred[chain_type].append(float(row["Predicted Label"]))
#            try:
#                chain_labels_true[chain_type].append(float(row["True Label"]))
#                chain_labels_pred[chain_type].append(float(row["Predicted Label"]))
#            except ValueError:
#                chain_labels_true[chain_type].append(0.0)
#                chain_labels_pred[chain_type].append(0.0)
        else:
            # Skip unrecognized chain types.
            continue

print("Labels per chain type:")
for ct in chain_labels_true:
    print(f"  {ct}: {len(chain_labels_true[ct])} residues")

# Read the original PDB file.
with open(pdb_filename, "r") as f:
    pdb_lines = f.readlines()

# Prepare lists for the modified PDB lines.
true_pdb_lines = []
pred_pdb_lines = []

# Mapping from PDB chain ID to our chain type.
chain_mapping = {"L": "Lchain", "H": "Hchain", "A": "AGchain_0"}

# For each chain type, maintain a counter and the last seen residue identifier.
residue_counters = {"Lchain": -1, "Hchain": -1, "AGchain_0": -1}
last_residue_ids = {"Lchain": None, "Hchain": None, "AGchain_0": None}

# Process each line of the PDB file.
# Occupancy field is in columns 55-60 (Python indices 54:60).
for line in pdb_lines:
    if line.startswith("ATOM"):
        # Extract the chain ID (column 22) and residue sequence number (columns 23-26).
        chain_id = line[21].strip()           # Column 22 (index 21)
        res_seq = line[22:26].strip()           # Columns 23-26 (indices 22-26)
        residue_id = (chain_id, res_seq)
        
        # Determine chain type based on the mapping.
        chain_type = chain_mapping.get(chain_id, None)
        
        if chain_type is not None:
            # Check if this line starts a new residue for that chain.
            if residue_id != last_residue_ids[chain_type]:
                residue_counters[chain_type] += 1
                last_residue_ids[chain_type] = residue_id
            
            idx = residue_counters[chain_type]
#            print(idx)
#            print(len(chain_labels_true[chain_type]))
            if idx < len(chain_labels_true[chain_type]):
                true_val = chain_labels_true[chain_type][idx]
                pred_val = chain_labels_pred[chain_type][idx]
            else:
                true_val = 0.0
                pred_val = 0.0
            
            # Format the occupancy value as a 6-character field with 2 decimals.
            new_true_occ = f"{true_val:6.2f}"
            new_pred_occ = f"{pred_val:6.2f}"
            
            # Replace the occupancy field (columns 55-60, i.e., indices 54:60).
            new_true_line = line[:54] + new_true_occ + line[60:]
            new_pred_line = line[:54] + new_pred_occ + line[60:]
        else:
            # If the chain is not recognized, leave the line unchanged.
            new_true_line = line
            new_pred_line = line
        
        true_pdb_lines.append(new_true_line)
        pred_pdb_lines.append(new_pred_line)
    else:
        # Write non-ATOM lines unchanged.
        true_pdb_lines.append(line)
        pred_pdb_lines.append(line)

# Define output filenames by appending _true and _pred before the .pdb extension.
true_pdb_filename = pdb_filename.replace(".pdb", "_true.pdb")
pred_pdb_filename = pdb_filename.replace(".pdb", "_pred.pdb")

# Save the modified PDB files.
with open(true_pdb_filename, "w") as f:
    f.writelines(true_pdb_lines)

with open(pred_pdb_filename, "w") as f:
    f.writelines(pred_pdb_lines)

print(f"Saved true-label PDB as: {true_pdb_filename}")
print(f"Saved predicted-label PDB as: {pred_pdb_filename}")

