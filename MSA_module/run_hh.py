import os
import subprocess

def run_hhblits(query_fasta, output_hhr, output_a3m, db_path, n_iterations=3, e_value=0.001, num_threads=4):
    """
    Runs hhblits on a query sequence against the uniref_30_2023_02 database.

    Args:
    - query_fasta: Path to the input query FASTA file
    - output_hhr: Path to the output HHR file (results)
    - db_path: Path to the uniref_30_2023_02 database
    - n_iterations: Number of hhblits iterations (default 3)
    - e_value: E-value threshold (default 0.001)
    - num_threads: Number of threads to use (default 4)
    """
    cmd = [
        "hhblits",
        "-i", query_fasta,          # Input query FASTA file
        "-o", output_hhr,           # Output HHR file
        "-oa3m", output_a3m,        # Output a3m file
        "-d", db_path,              # Path to uniref_30_2023_02 database
        "-n", str(n_iterations),    # Number of iterations
        "-e", str(e_value),         # E-value threshold
        "-cpu", str(num_threads)    # Number of CPU threads
    ]

    print("Running HHblits with command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"HHblits search completed. Results saved to {output_hhr}")

# Example usage:
query_fasta = "input_query.fasta"  # Path to your input FASTA file
output_hhr = "output_results.hhr"  # Path to the output HHR file
db_path = "/path/to/uniref_30_2023_02"  # Path to the uniref_30_2023_02 database files

# Run the hhblits search
run_hhblits(query_fasta, output_hhr, output_a3m, db_path)

