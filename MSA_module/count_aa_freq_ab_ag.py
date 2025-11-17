import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ESM token → AA map (20 canonical + special)
ID2AA = {
    0:"CLS", 1:"PAD", 2:"EOS", 3:"UNK",
    4:"L", 5:"A", 6:"G", 7:"V", 8:"S", 9:"E", 10:"R", 11:"T", 12:"I",
    13:"D", 14:"P", 15:"K", 16:"Q", 17:"N", 18:"F", 19:"Y", 20:"M",
    21:"H", 22:"W", 23:"C", 24:"X", 25:"B", 26:"U", 27:"Z", 28:"O", 29:".", 30:"-"
}

CANONICAL = ["A","R","N","D","C","Q","E","G","H","I","L",
             "K","M","F","P","S","T","W","Y","V"]
CANONICAL_SET = set(CANONICAL)

def main():
    ap = argparse.ArgumentParser(description="Compute AA frequency in antibody & antigen from sequence NPZ")
    ap.add_argument("--seq-npz", required=True, type=Path, help="Sequence NPZ file used for test set")
    ap.add_argument("--out", default="aa_freq_ab_ag.csv", type=Path)
    args = ap.parse_args()

    z = np.load(args.seq_npz, allow_pickle=True)

    ab_counts = {aa:0 for aa in CANONICAL}
    ag_counts = {aa:0 for aa in CANONICAL}

    ab_total = 0
    ag_total = 0
    skipped = 0

    for pdbid in z.files:
        seq = z[pdbid][0]   # first row = real sequence tokens
        tokens = np.array(seq, dtype=int)

        # Find EOC tokens separating chains
        eocs = np.where(tokens == 24)[0]
        if len(eocs) < 2:
            skipped += 1
            continue

        L_end = eocs[0]
        H_end = eocs[1]

        # Define masks for antibody (L+H) and antigen
        ab_mask = np.zeros_like(tokens, dtype=bool)
        ab_mask[:L_end] = True
        ab_mask[L_end+1:H_end] = True

        ag_mask = np.zeros_like(tokens, dtype=bool)
        ag_mask[H_end+1:] = True

        # Convert tokens to AAs and count
        aa_all = np.array([ID2AA[t] for t in tokens])

        ab_aa = aa_all[ab_mask]
        ag_aa = aa_all[ag_mask]

        for aa in ab_aa:
            if aa in CANONICAL_SET:
                ab_counts[aa] += 1
                ab_total += 1

        for aa in ag_aa:
            if aa in CANONICAL_SET:
                ag_counts[aa] += 1
                ag_total += 1

    # Build results
    df = pd.DataFrame({
        "AA": CANONICAL,
        "Antibody_Count": [ab_counts[a] for a in CANONICAL],
        "Antigen_Count": [ag_counts[a] for a in CANONICAL],
        "Antibody_Freq": [ab_counts[a]/ab_total if ab_total>0 else 0 for a in CANONICAL],
        "Antigen_Freq": [ag_counts[a]/ag_total if ag_total>0 else 0 for a in CANONICAL],
    })

    df.to_csv(args.out, index=False)
    print(f"[OK] Saved amino-acid frequency table → {args.out}")
    print(f"Structures skipped due to missing EOCs: {skipped}")

if __name__ == "__main__":
    main()

