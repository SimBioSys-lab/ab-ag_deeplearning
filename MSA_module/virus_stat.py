#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def read_metrics(metrics_csv: Path, subset: str = "ALL") -> pd.DataFrame:
    """
    Load metrics and return per-PDB AUC-PR for a chosen subset.
    - If 'subset' column exists, filter by the requested subset (ALL/AB/AG).
    - If 'subset' column does not exist, proceed as legacy (no split).
    Keeps only rows where PDBID looks like a 4-char code.
    """
    df = pd.read_csv(metrics_csv)
    df.columns = [c.strip() for c in df.columns]

    # Basic validation
    if "PDBID" not in df.columns or "auc_pr" not in df.columns:
        raise ValueError("metrics CSV must include columns: PDBID, auc_pr")

    # If split metrics are present, filter to requested subset
    if "subset" in df.columns:
        if subset is None:
            # No filtering; keep all subsets (used when caller wants all)
            pass
        else:
            # Normalize subset tokens to upper
            subset_norm = subset.strip().upper()
            df["subset"] = df["subset"].astype(str).str.upper().str.strip()
            # Keep only the requested subset
            df = df.loc[df["subset"] == subset_norm].copy()

    # Keep only PDB-like rows (4-char codes), ignore pooled ALL/MACRO rows
    pdf = df.loc[df["PDBID"].astype(str).str.len() == 4].copy()
    pdf["PDBID"] = pdf["PDBID"].astype(str).str.upper().str.strip()

    # First occurrence wins if duplicates
    pdf = pdf.drop_duplicates(subset=["PDBID"], keep="first")
    pdf["auc_pr"] = pd.to_numeric(pdf["auc_pr"], errors="coerce")

    # Return only what summarize() needs
    cols = ["PDBID", "auc_pr"]
    if "subset" in df.columns:
        # Keep subset for possible downstream debugging/details
        pdf["subset"] = subset if subset is not None else pdf.get("subset", np.nan)
        cols.append("subset")

    return pdf[cols]


def read_virus_book(xlsx_path: Path) -> Dict[str, pd.Series]:
    """
    Return {virus_name: unique_uppercase_pdb_series}
    """
    book = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    out: Dict[str, pd.Series] = {}
    for sheet, df in book.items():
        cols = [c.strip() for c in df.columns]
        if "PDB_Code" not in cols:
            raise ValueError(f"Sheet '{sheet}' missing required column 'PDB_Code'")
        pdbs = (
            df["PDB_Code"]
            .astype(str)
            .str.upper()
            .str.strip()
            .dropna()
        )
        pdbs = pd.Series(pd.unique(pdbs), name="PDBID")  # unique, keep order
        out[sheet] = pdbs
    return out


def summarize(virus_to_pdbs: Dict[str, pd.Series], metrics: pd.DataFrame
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (summary_df, detail_df) for the already-filtered metrics set.
    """
    detail_rows: List[dict] = []
    summary_rows: List[dict] = []

    for virus, pdb_series in virus_to_pdbs.items():
        listed = pdb_series.tolist()

        # Join to fetch auc_pr for listed PDBs
        sub = pd.DataFrame({"PDBID": listed}).merge(metrics, on="PDBID", how="left")

        # stats (drop NaN auc_pr)
        valid = sub["auc_pr"].dropna()
        row = {
            "virus": virus,
            "n_listed": len(listed),
            "n_found": int(sub["auc_pr"].notna().sum()),
            "n_missing": int(sub["auc_pr"].isna().sum()),
            "n_valid_aucpr": int(valid.shape[0]),
            "mean_aucpr": float(valid.mean()) if not valid.empty else np.nan,
            "median_aucpr": float(valid.median()) if not valid.empty else np.nan,
            "std_aucpr": float(valid.std(ddof=1)) if valid.shape[0] >= 2 else np.nan,
            "min_aucpr": float(valid.min()) if not valid.empty else np.nan,
            "max_aucpr": float(valid.max()) if not valid.empty else np.nan,
        }
        summary_rows.append(row)

        # detail rows
        sub_detail = sub.copy()
        sub_detail.insert(0, "virus", virus)
        sub_detail["missing_aucpr"] = sub_detail["auc_pr"].isna()
        # Keep subset column if present in metrics
        if "subset" in metrics.columns and "subset" not in sub_detail.columns:
            sub_detail["subset"] = metrics.get("subset", np.nan)
        detail_rows.extend(sub_detail.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)[
        ["virus", "n_listed", "n_found", "n_missing", "n_valid_aucpr",
         "mean_aucpr", "median_aucpr", "std_aucpr", "min_aucpr", "max_aucpr"]
    ].sort_values("virus")

    # Detail view columns (include subset if present)
    detail_df = pd.DataFrame(detail_rows)
    cols = ["virus", "PDBID", "auc_pr", "missing_aucpr"]
    if "subset" in detail_df.columns:
        cols.insert(2, "subset")
    detail_df = detail_df[cols]

    return summary_df, detail_df


def main():
    ap = argparse.ArgumentParser(
        description="Summarize AUC-PR per virus from Excel sheets + metrics CSV (supports subset splits)."
    )
    ap.add_argument("--xlsx", required=True, type=Path, help="Excel workbook with 1 sheet per virus")
    ap.add_argument("--metrics", required=True, type=Path, help="calc_pdb_metrics(_split).py output CSV")
    ap.add_argument("--out-summary", required=True, type=Path, help="Output CSV for per-virus summary")
    ap.add_argument("--out-detail", required=True, type=Path, help="Output CSV for per-virus per-PDB details")
    ap.add_argument(
        "--subset",
        choices=["ALL", "AB", "AG"],
        default="ALL",
        help="Which subset to summarize when metrics contain a 'subset' column (default: ALL). "
             "Ignored if metrics have no 'subset' column."
    )
    args = ap.parse_args()

    if not args.xlsx.exists():
        raise FileNotFoundError(f"Excel not found: {args.xlsx}")
    if not args.metrics.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {args.metrics}")

    # Load metrics filtered to the chosen subset (or legacy behavior)
    metrics = read_metrics(args.metrics, subset=args.subset)
    virus_to_pdbs = read_virus_book(args.xlsx)

    summary_df, detail_df = summarize(virus_to_pdbs, metrics)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_detail.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)
    detail_df.to_csv(args.out_detail, index=False)

    # Pretty console print
    pd.set_option("display.float_format", lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
    print(f"\n=== Per-virus AUC-PR summary (subset={args.subset}) ===")
    print(summary_df.to_string(index=False))
    print(f"\n[OK] Wrote summary -> {args.out_summary}")
    print(f"[OK] Wrote detail  -> {args.out_detail}")


if __name__ == "__main__":
    main()

