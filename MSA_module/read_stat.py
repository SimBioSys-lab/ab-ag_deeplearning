#!/usr/bin/env python3
"""
Compute AUC-ROC and AUC-PR per PDB and overall from residue-level predictions,
with per-subset splits for Antibody (L/H chains) and Antigen (AGchain_#).

Assumptions (same as before):
- CSV contains multiple PDBs concatenated.
- A new PDB block starts at any row where 'Chain Type' starts with 'L' (case-insensitive)
  AND (it's the first row OR the previous row does NOT start with 'L').
- NPZ keys are exactly the PDB IDs. We map CSV block i -> NPZ key i by default,
  or alphabetically with --order alpha.

Usage:
  python calc_pdb_metrics_split.py --csv preds.csv --npz pdb_index.npz --out metrics.csv
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def pdb_ids_from_npz(npz_path: Path, order: str = "npz") -> List[str]:
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.keys())
    if not keys:
        raise ValueError(f"No keys found in NPZ: {npz_path}")
    return [k.upper() for k in (sorted(keys) if order == "alpha" else keys)]


def find_pdb_blocks(chain_series: pd.Series) -> List[Tuple[int, int]]:
    vals = chain_series.astype(str).str.lower().str.strip().values
    starts = []
    for i, c in enumerate(vals):
        if c.startswith("l") and (i == 0 or not vals[i - 1].startswith("l")):
            starts.append(i)
    if not starts:
        raise ValueError("No PDB starts detected. Ensure 'Chain Type' has 'L...' markers.")
    blocks = []
    for j, s in enumerate(starts):
        e = (starts[j + 1] - 1) if (j + 1 < len(starts)) else (len(vals) - 1)
        blocks.append((s, e))
    return blocks


def read_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    expected = {"Chain Type", "True Label", "Predicted Label", "Probability"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")
    df["Chain Type"] = df["Chain Type"].astype(str)
    df["True Label"] = pd.to_numeric(df["True Label"], errors="coerce").astype("Int64")
    df["Predicted Label"] = pd.to_numeric(df["Predicted Label"], errors="coerce").astype("Int64")
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    return df


def safe_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    uniq = np.unique(y_true[~np.isnan(y_true)])
    if uniq.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    uniq = np.unique(y_true[~np.isnan(y_true)])
    if uniq.size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def compute_metrics_block(df_block: pd.DataFrame) -> Dict[str, Any]:
    y_true = df_block["True Label"].astype(int).to_numpy()
    y_score = df_block["Probability"].to_numpy(dtype=float)
    n = len(df_block)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc_roc": safe_auc_roc(y_true, y_score),
        "auc_pr": auc_pr(y_true, y_score),
    }


def subset_masks(chain_series: pd.Series) -> Dict[str, pd.Series]:
    """Return boolean masks for subsets within a block."""
    s = chain_series.astype(str).str.lower().str.strip()
    mask_ab = s.str.startswith("l") | s.str.startswith("h")     # Antibody: L or H
    mask_ag = s.str.startswith("ag")                            # Antigen: AGchain_#
    mask_all = pd.Series(True, index=s.index)
    return {"ALL": mask_all, "AB": mask_ab, "AG": mask_ag}


def add_macro_rows(rows: List[Dict[str, Any]], label: str) -> None:
    """Append a MACRO row across PDBs for a given subset label (ALL/AB/AG)."""
    per = [r for r in rows if r["subset"] == label and r["PDBID"] not in ("ALL", "MACRO_ALL", "MACRO_AB", "MACRO_AG")]
    if not per:
        return
    auc_roc_vals = pd.Series([r["auc_roc"] for r in per], dtype=float)
    auc_pr_vals = pd.Series([r["auc_pr"] for r in per], dtype=float)
    rows.append({
        "PDBID": f"MACRO_{label}",
        "subset": label,
        "n": int(sum(r["n"] for r in per)),
        "n_pos": int(sum(r["n_pos"] for r in per)),
        "n_neg": int(sum(r["n_neg"] for r in per)),
        "auc_roc": float(auc_roc_vals.mean(skipna=True)),
        "auc_pr": float(auc_pr_vals.mean(skipna=True)),
    })


def main():
    ap = argparse.ArgumentParser(description="Compute per-PDB and per-subset (AB/AG) AUC-ROC/AUC-PR.")
    ap.add_argument("--csv", required=True, type=Path, help="Predictions CSV")
    ap.add_argument("--npz", required=True, type=Path, help="NPZ whose KEYS are PDB IDs")
    ap.add_argument("--out", type=Path, default=None, help="Output metrics CSV (default: <csv stem>_metrics.csv)")
    ap.add_argument("--order", choices=["npz", "alpha"], default="npz",
                    help="Order in which to read NPZ keys (default: npz)")
    ap.add_argument("--strict", action="store_true",
                    help="Error if #CSV blocks != #NPZ PDB IDs (default: align by min count).")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.npz.exists():
        raise FileNotFoundError(f"NPZ not found: {args.npz}")

    df = read_predictions(args.csv)
    blocks = find_pdb_blocks(df["Chain Type"])
    pdb_ids = pdb_ids_from_npz(args.npz, order=args.order)

    n_blocks, n_ids = len(blocks), len(pdb_ids)
    if args.strict and n_blocks != n_ids:
        raise ValueError(f"Block/ID count mismatch: CSV blocks={n_blocks}, NPZ IDs={n_ids}")
    n_map = min(n_blocks, n_ids)
    if n_map < n_blocks or n_map < n_ids:
        print(f"[WARN] Mapping first {n_map} blocks to {n_map} PDB IDs "
              f"(CSV blocks={n_blocks}, NPZ IDs={n_ids}).")

    rows = []
    # Per-PDB metrics (ALL / AB / AG)
    for i in range(n_map):
        s, e = blocks[i]
        pid = pdb_ids[i]
        block_df = df.iloc[s:e+1].copy()

        masks = subset_masks(block_df["Chain Type"])
        for label, mask in masks.items():
            sub_df = block_df[mask]
            if sub_df.empty:
                # If a subset doesn't exist in this PDB block, still record NaNs for clarity
                rows.append({"PDBID": pid, "subset": label, "n": 0, "n_pos": 0, "n_neg": 0,
                             "auc_roc": float("nan"), "auc_pr": float("nan")})
            else:
                metrics = compute_metrics_block(sub_df)
                rows.append({"PDBID": pid, "subset": label, **metrics})

    # Overall metrics (ALL rows up to last mapped block)
    end_idx = blocks[n_map-1][1] if n_map > 0 else len(df) - 1
    overall_df = df.iloc[:end_idx+1] if end_idx >= 0 else df.iloc[0:0]

    overall_masks = subset_masks(overall_df["Chain Type"])
    for label, mask in overall_masks.items():
        sub_df = overall_df[mask]
        if sub_df.empty:
            rows.append({"PDBID": "ALL", "subset": label, "n": 0, "n_pos": 0, "n_neg": 0,
                         "auc_roc": float("nan"), "auc_pr": float("nan")})
        else:
            metrics = compute_metrics_block(sub_df)
            rows.append({"PDBID": "ALL", "subset": label, **metrics})

    # Macro-averages across PDBs for each subset
    add_macro_rows(rows, "ALL")
    add_macro_rows(rows, "AB")
    add_macro_rows(rows, "AG")

    out_path = args.out or args.csv.with_name(args.csv.stem + "_metrics.csv")
    out_df = pd.DataFrame(rows, columns=["PDBID", "subset", "n", "n_pos", "n_neg", "auc_roc", "auc_pr"])
    out_df.to_csv(out_path, index=False)

    # Pretty print summary
    print(f"[OK] Wrote metrics: {out_path}")
    def ffmt(x):
        return f"{x:.6f}" if isinstance(x, float) else str(x)
    print(out_df.to_string(index=False, justify='left', float_format=ffmt))


if __name__ == "__main__":
    main()

