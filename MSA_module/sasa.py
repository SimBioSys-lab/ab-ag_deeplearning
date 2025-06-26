import os
import pandas as pd
import freesasa

def calculate_sasa(pdb_code, chain_id, sasa_data):
    """
    Calculate SASA for a given PDB chain using FreeSASA.

    Args:
        pdb_code (str): PDB code of the structure.
        chain_id (str): Chain identifier.
        sasa_data (list): List to store the results.
    """
    pdb_filename = f"/work/SimBioSys/Xing/data_collection/pdbs/pdb_chains/{pdb_code}_chain_{chain_id}.pdb"
    if not os.path.exists(pdb_filename):
        print(f"File not found: {pdb_filename}")
        return

    try:
        # Calculate SASA using FreeSASA
        structure = freesasa.Structure(pdb_filename)
        result = freesasa.calc(structure)

        # Collect SASA data for each residue
        for i in range(structure.nAtoms()):
            atom = structure.atomName(i)
            res_name = structure.residueName(i)
            res_number = structure.residueNumber(i)
            chain = structure.chainLabel(i)
            sasa = result.atomArea(i)

            # Append to the SASA data list
            sasa_data.append([pdb_code, chain, res_number, res_name, sasa])

    except Exception as e:
        print(f"Error calculating SASA for {pdb_code}, Chain {chain_id}: {e}")


# Load data from Excel file
abag_data = pd.read_excel("final_ab-ag_summary_short.xlsx", sheet_name="Sheet1")

# Validate required columns
expected_columns = [
    "PDB_Code", "Hchain", "Lchain", "AGchain", "compound", "species",
    "short_header", "antigen", "method", "affinity", "deltaG", "aff_method"
]

missing_columns = [col for col in expected_columns if col not in abag_data.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

# Collect all SASA data
sasa_data = []

# Process each row in the dataset
for _, row in abag_data.iterrows():
    pdb_code = row['PDB_Code'].lower()

    # Check if required chains are present
    if pd.isna(row['Hchain']) or pd.isna(row['Lchain']) or pd.isna(row['AGchain']):
        print(f"Skipping {pdb_code}: Missing chain data.")
        continue

    # Process heavy and light chains
    Hchain = row['Hchain']
    Lchain = row['Lchain']

    calculate_sasa(pdb_code, Hchain, sasa_data)
    calculate_sasa(pdb_code, Lchain, sasa_data)

    # Process antigen chains
    AGchains = [ag.strip() for ag in str(row['AGchain']).split('|') if ag.strip()]
    for chain in AGchains:
        calculate_sasa(pdb_code, chain, sasa_data)

# Convert the SASA data to a DataFrame
sasa_df = pd.DataFrame(
    sasa_data,
    columns=["PDB_Code", "Chain", "Residue_Number", "Residue_Name", "SASA"]
)

# Save to a CSV file
sasa_csv_filename = "sasa_atom_data.csv"
sasa_df.to_csv(sasa_csv_filename, index=False)
print(f"SASA data saved to {sasa_csv_filename}")

# Aggregate SASA values by residue
residue_sasa_df = sasa_df.groupby(["PDB_Code", "Chain", "Residue_Number", "Residue_Name"], as_index=False).agg({"SASA": "sum"})

# Save the aggregated SASA data to a new CSV file
residue_sasa_df.to_csv("sasa_data.csv", index=False)

print("Residue-level SASA data saved to sasa_data.csv")

