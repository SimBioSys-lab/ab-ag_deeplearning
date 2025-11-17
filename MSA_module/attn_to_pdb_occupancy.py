#!/usr/bin/env python3
"""
Set PDB residue occupancies from last-layer multi-head attention.

Assumes attention CSVs are named:  <PDBID>__head0.csv ... <PDBID>__head15.csv
Each CSV is an NxN matrix (e.g., 1600x1600) for that head.

We:
  1) Load all heads for the given PDBID, average across heads -> A (N x N)
  2) Reduce per-token score s[i] from A using a chosen mode:
       - row_mean, col_mean, row_max, col_max, sym_mean, sym_max (default)
  3) Map tokens -> residues by chains you provide, skipping EOC tokens between chains
  4) Write a new PDB with residue occupancies set to s (per-residue, applied to all atoms)

Usage:
  python attn_to_pdb_occupancy.py \
    --pdbid 7LBE \
    --pdb-in 7LBE.pdb --pdb-out 7LBE_attn_occ.pdb \
    --chains H L C \
    --attn-glob "7LBE__head*.csv" \
    --reduce sym_max \
    --eoc 1 \
    --norm percentile --p_lo 1 --p_hi 99 \
    --map-out 7LBE_attn_occ_map.csv

Requires: biopython, pandas, numpy, matplotlib (optional)
"""

import argparse, glob, os, re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, PDBIO, Select


def load_head_csv(path: str) -> np.ndarray:
    """Load a single head CSV as float32 (NxN)."""
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = pd.read_csv(path, header=None).values
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{path}: expected square matrix, got {arr.shape}")
    return arr


def load_attn_stack(attn_glob: str) -> Tuple[np.ndarray, List[str]]:
    """Load all heads matching glob; sort by head index if present."""
    files = glob.glob(attn_glob)
    if not files:
        raise SystemExit(f"No attention files match: {attn_glob}")
    def head_idx(fn: str) -> int:
        m = re.search(r"__head(\d+)\.csv$", fn)
        return int(m.group(1)) if m else 10**9  # unsuffixed at end
    files = sorted(files, key=head_idx)
    mats = [load_head_csv(f) for f in files]
    # sanity: all same size
    nset = {M.shape for M in mats}
    if len(nset) != 1:
        raise ValueError(f"Attention files have mixed shapes: {nset}")
    stack = np.stack(mats, axis=0)  # (H, N, N)
    return stack, files


def reduce_token_scores(A: np.ndarray, mode: str = "sym_max") -> np.ndarray:
    """
    A: (N,N) attention averaged over heads (row=query, col=key)
    Returns s: (N,) per-token score.
    """
    if mode == "row_mean":
        s = A.mean(axis=1)
    elif mode == "col_mean":
        s = A.mean(axis=0)
    elif mode == "row_max":
        s = A.max(axis=1)
    elif mode == "col_max":
        s = A.max(axis=0)
    elif mode == "sym_mean":
        s = 0.5 * (A.mean(axis=1) + A.mean(axis=0))
    elif mode == "sym_max":
        s = 0.5 * (A.max(axis=1) + A.max(axis=0))
    else:
        raise ValueError(f"Unknown reduce mode: {mode}")
    return s.astype(np.float32)


def normalize_scores(s: np.ndarray, how: str = "none",
                     p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Normalize to [0,1] optionally, to fit occupancy expectations."""
    if how == "none":
        return s
    if how == "minmax":
        lo, hi = float(s.min()), float(s.max())
    elif how == "percentile":
        lo, hi = np.percentile(s, [p_lo, p_hi]).astype(float)
    else:
        raise ValueError(f"Unknown norm: {how}")
    if hi <= lo:
        return np.zeros_like(s) + 0.0
    out = (s - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def get_chain_residues(pdb_path: str, pdbid: str, chain_ids: List[str]):
    """
    Return (structure, {chain_id: [Residue,...]}) using Biopython.
    Only standard residues (hetflag ' ') are included.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdbid, pdb_path)
    model = next(structure.get_models())
    chain_map = {}
    for cid in chain_ids:
        chain = model[cid]
        residues = [res for res in chain.get_residues() if res.id[0] == ' ']
        chain_map[cid] = residues
    return structure, chain_map


def assign_scores_to_chains(s: np.ndarray, chain_map: dict, chain_order: List[str],
                            eoc: int = 1) -> List[Tuple[str, int, float]]:
    """
    Map token scores s onto residues in the given chain order, skipping `eoc` tokens between chains.
    Returns a flat list of (chain_id, local_idx, score) with local_idx in [0, len(chain)-1].
    """
    out = []
    idx = 0
    for k, cid in enumerate(chain_order):
        residues = chain_map[cid]
        L = len(residues)
        if idx + L > len(s):
            raise ValueError(f"Token vector too short: need {idx+L}, have {len(s)}. "
                             f"(chain {cid} has {L} residues)")
        seg = s[idx: idx + L]
        out.extend((cid, j, float(seg[j])) for j in range(L))
        idx += L
        if k < len(chain_order) - 1:
            idx += eoc  # skip EOC between chains
    return out


def set_occupancies(structure, chain_map: dict,
                    assignments: List[Tuple[str, int, float]]) -> List[dict]:
    """
    Set occupancy for all atoms in each residue according to assigned score.
    Returns a list of mapping rows for CSV: chain, resseq, icode, resname, score.
    """
    rows = []
    for cid, local_idx, val in assignments:
        res = chain_map[cid][local_idx]
        # set for all atoms in residue
        for atom in res.get_atoms():
            atom.set_occupancy(float(val))
        resseq = res.id[1]
        icode = res.id[2].strip() if isinstance(res.id[2], str) else res.id[2]
        rows.append({
            "chain": cid,
            "local_idx": local_idx,
            "resseq": resseq,
            "icode": icode,
            "resname": res.get_resname(),
            "occupancy": float(val),
        })
    return rows


class KeepAll(Select):
    def accept_atom(self, atom):
        return 1


def main():
    ap = argparse.ArgumentParser(description="Write PDB occupancies from attention matrices.")
    ap.add_argument("--pdbid", required=True, help="PDB code (used for structure ID and file naming)")
    ap.add_argument("--pdb-in", required=True, help="Input PDB file path")
    ap.add_argument("--pdb-out", required=True, help="Output PDB file path with occupancies set")
    ap.add_argument("--chains", nargs="+", required=True, help="Chain IDs in token order, e.g. H L C")
    ap.add_argument("--attn-glob", required=True, help="Glob for per-head CSVs, e.g. '7LBE__head*.csv'")
    ap.add_argument("--reduce", default="sym_max",
                    choices=["row_mean","col_mean","row_max","col_max","sym_mean","sym_max"],
                    help="How to reduce NxN to per-token score (default: sym_max)")
    ap.add_argument("--eoc", type=int, default=1, help="Number of EOC tokens to skip between chains (default 1)")
    ap.add_argument("--norm", default="none", choices=["none","minmax","percentile"],
                    help="Optional normalization of per-token scores to [0,1]")
    ap.add_argument("--p_lo", type=float, default=1.0, help="Lower percentile for --norm percentile")
    ap.add_argument("--p_hi", type=float, default=99.0, help="Upper percentile for --norm percentile")
    ap.add_argument("--map-out", default=None, help="Optional CSV mapping residues to assigned occupancy")
    args = ap.parse_args()

    # 1) Load attention, average heads
    stack, files = load_attn_stack(args.attn_glob)     # (H,N,N)
    A = stack.mean(axis=0)                              # (N,N)
    print(f"[INFO] Loaded {len(files)} heads, matrix size {A.shape}")

    # 2) Reduce to per-token scores and (optionally) normalize for occupancy [0,1]
    s = reduce_token_scores(A, mode=args.reduce)        # (N,)
    s = normalize_scores(s, how=args.norm, p_lo=args.p_lo, p_hi=args.p_hi)
    print(f"[INFO] Token scores: min={s.min():.4f}, mean={s.mean():.4f}, max={s.max():.4f}")

    # 3) Read PDB and chains
    structure, chain_map = get_chain_residues(args.pdb_in, args.pdbid, args.chains)
    for cid in args.chains:
        print(f"  - Chain {cid}: {len(chain_map[cid])} residues")

    # 4) Assign scores to residues in chain order, skipping EOC tokens
    assigns = assign_scores_to_chains(s, chain_map, args.chains, eoc=args.eoc)
    total_res = sum(len(chain_map[c]) for c in args.chains)
    print(f"[INFO] Assigned scores to {len(assigns)}/{total_res} residues across {len(args.chains)} chains.")

    # 5) Write occupancies into PDB
    rows = set_occupancies(structure, chain_map, assigns)
    io = PDBIO()
    io.set_structure(structure)
    Path(args.pdb_out).parent.mkdir(parents=True, exist_ok=True)
    io.save(args.pdb_out, select=KeepAll())
    print(f"[OK] Wrote modified PDB -> {args.pdb_out}")

    # 6) Optional residue→score map
    if args.map_out:
        df = pd.DataFrame(rows, columns=["chain","local_idx","resseq","icode","resname","occupancy"])
        Path(args.map_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.map_out, index=False)
        print(f"[OK] Wrote residue map -> {args.map_out}")


if __name__ == "__main__":
    main()

