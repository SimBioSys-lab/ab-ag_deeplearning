#!/usr/bin/env python3
"""
Subset an NPZ by PDB IDs listed in CSV files (header=None).

- Input NPZ: para_tv_esmsequences_1600.npz (keys are lowercase PDB IDs)
- Input CSVs (header=None): rows like "1A3R,L,H,P" (only first column used)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_pdb_ids(csv_paths):
    """Read first column from CSVs (header=None), return unique lowercase IDs."""
    ids = []
    for p in csv_paths:
        df = pd.read_csv(p, header=None, dtype=str)
        if 0 not in df.columns:
            continue
        col = (
            df[0]
            .astype(str)
            .str.strip()
            .pipe(lambda s: s[~s.str.startswith("#", na=False)])  # drop comment rows
            .replace({"": np.nan})
            .dropna()
            .str.lower()
        )
        ids.append(col)
    if not ids:
        return set()
    return set(pd.concat(ids, ignore_index=True).unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="para_tv_esmsequences_1600.npz",
                    help="Input NPZ (keys: lowercase PDB IDs)")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="CSV file(s) with header=None; first column = PDB code")
    ap.add_argument("--out", default="subset_para_tv_esmsequences_1600.npz",
                    help="Output NPZ filename")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)
    csv_paths = [Path(p) for p in args.csv]

    wanted = load_pdb_ids(csv_paths)
    if not wanted:
        raise SystemExit("No PDB IDs found in provided CSV(s).")

    data = np.load(npz_path, allow_pickle=True)
    have = set(data.files)

    selected = sorted(wanted & have)
    missing  = sorted(wanted - have)

    if not selected:
        raise SystemExit("None of the requested PDB IDs were found in the NPZ.")

    subset = {k: data[k] for k in selected}
    np.savez_compressed(out_path, **subset)

    print(f"Input:  {npz_path}")
    print(f"CSVs:   {', '.join(str(p) for p in csv_paths)}")
    print(f"Found:  {len(selected)}/{len(wanted)} IDs")
    print(f"Saved:  {out_path}")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))


if __name__ == "__main__":
    main()

