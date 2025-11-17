#!/usr/bin/env python3
"""
Set PDB residue B-factors from last-layer multi-head attention.

Assumes attention CSVs are named:  <PDBID>__head0.csv ... <PDBID>__head15.csv
Each CSV is an NxN matrix (e.g., 1600x1600) for that head.

Steps:
  1) Load all heads for the PDB, average -> A (N x N)
  2) Reduce per-token score s[i] from A (row/col mean/max or symmetric)  -> s (N,)
  3) Optionally normalize s to [0,1] (minmax / percentile)
  4) Rescale: B = bf_offset + bf_scale * s   (defaults: 0 + 100*s)
  5) Map tokens -> residues across the given chain order, skipping EOC tokens between chains
  6) Write a PDB with atom B-factors set to B (per residue)

Usage:
  python attn_to_pdb_bfactor.py \
    --pdbid 7LBE \
    --pdb-in 7LBE.pdb --pdb-out 7LBE_bfac.pdb \
    --chains H L C \
    --attn-glob "7LBE__head*.csv" \
    --reduce sym_max \
    --eoc 1 \
    --norm percentile --p_lo 5 --p_hi 99 \
    --bf-scale 100 --bf-offset 0 \
    --map-out 7LBE_bfac_map.csv

Requires: biopython, pandas, numpy
"""

import argparse, glob, re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select


def load_head_csv(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = pd.read_csv(path, header=None).values
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{path}: expected square matrix, got {arr.shape}")
    return arr


def load_attn_stack(attn_glob: str) -> Tuple[np.ndarray, List[str]]:
    files = glob.glob(attn_glob)
    if not files:
        raise SystemExit(f"No attention files match: {attn_glob}")
    def head_idx(fn: str) -> int:
        m = re.search(r"__head(\d+)\.csv$", fn)
        return int(m.group(1)) if m else 10**9
    files = sorted(files, key=head_idx)
    mats = [load_head_csv(f) for f in files]
    shapes = {M.shape for M in mats}
    if len(shapes) != 1:
        raise ValueError(f"Attention files have mixed shapes: {shapes}")
    stack = np.stack(mats, axis=0)  # (H, N, N)
    return stack, files


def reduce_token_scores(A: np.ndarray, mode: str = "sym_max") -> np.ndarray:
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
    if how == "none":
        return s
    if how == "minmax":
        lo, hi = float(s.min()), float(s.max())
    elif how == "percentile":
        lo, hi = [float(x) for x in np.percentile(s, [p_lo, p_hi])]
    else:
        raise ValueError(f"Unknown norm: {how}")
    if hi <= lo:
        return np.zeros_like(s)
    out = (s - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def get_chain_residues(pdb_path: str, pdbid: str, chain_ids: List[str]):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdbid, pdb_path)
    model = next(structure.get_models())
    chain_map = {}
    for cid in chain_ids:
        chain = model[cid]
        residues = [res for res in chain.get_residues() if res.id[0] == ' ']  # standard residues only
        chain_map[cid] = residues
    return structure, chain_map


def assign_scores_to_chains(s: np.ndarray, chain_map: dict, chain_order: List[str],
                            eoc: int = 1) -> List[Tuple[str, int, float]]:
    out = []
    idx = 0
    for k, cid in enumerate(chain_order):
        residues = chain_map[cid]
        L = len(residues)
        if idx + L > len(s):
            raise ValueError(f"Token vector too short: need {idx+L}, have {len(s)} (chain {cid} has {L})")
        seg = s[idx: idx + L]
        out.extend((cid, j, float(seg[j])) for j in range(L))
        idx += L
        if k < len(chain_order) - 1:
            idx += eoc  # skip EOC between chains
    return out


def set_bfactors(structure, chain_map: dict,
                 assignments: List[Tuple[str, int, float]],
                 bf_scale: float = 100.0, bf_offset: float = 0.0) -> List[dict]:
    rows = []
    for cid, local_idx, val in assignments:
        res = chain_map[cid][local_idx]
        B = float(bf_offset + bf_scale * val)
        for atom in res.get_atoms():
            atom.set_bfactor(B)
        resseq = res.id[1]
        icode = res.id[2].strip() if isinstance(res.id[2], str) else res.id[2]
        rows.append({
            "chain": cid,
            "local_idx": local_idx,
            "resseq": resseq,
            "icode": icode,
            "resname": res.get_resname(),
            "bfactor": B,
            "score": float(val),
        })
    return rows


class KeepAll(Select):
    def accept_atom(self, atom): return 1


def main():
    ap = argparse.ArgumentParser(description="Write PDB B-factors from attention matrices.")
    ap.add_argument("--pdbid", required=True)
    ap.add_argument("--pdb-in", required=True)
    ap.add_argument("--pdb-out", required=True)
    ap.add_argument("--chains", nargs="+", required=True, help="Chain IDs in token order, e.g. H L C")
    ap.add_argument("--attn-glob", required=True, help="Glob for per-head CSVs, e.g. '7LBE__head*.csv'")
    ap.add_argument("--reduce", default="sym_max",
                    choices=["row_mean","col_mean","row_max","col_max","sym_mean","sym_max"])
    ap.add_argument("--eoc", type=int, default=1, help="EOC tokens to skip between chains")
    ap.add_argument("--norm", default="none", choices=["none","minmax","percentile"])
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument("--bf-scale", type=float, default=100.0)
    ap.add_argument("--bf-offset", type=float, default=0.0)
    ap.add_argument("--map-out", default=None)
    args = ap.parse_args()

    # 1) attention -> token scores
    stack, files = load_attn_stack(args.attn_glob)
    A = stack.mean(axis=0)
    s = reduce_token_scores(A, mode=args.reduce)
    s = normalize_scores(s, how=args.norm, p_lo=args.p_lo, p_hi=args.p_hi)
    print(f"[INFO] Loaded {len(files)} heads, matrix {A.shape}, token score range {s.min():.4f}..{s.max():.4f}")

    # 2) pdb + chain map
    structure, chain_map = get_chain_residues(args.pdb_in, args.pdbid, args.chains)
    for cid in args.chains:
        print(f"  - Chain {cid}: {len(chain_map[cid])} residues")

    # 3) assign & write
    assigns = assign_scores_to_chains(s, chain_map, args.chains, eoc=args.eoc)
    rows = set_bfactors(structure, chain_map, assigns, bf_scale=args.bf_scale, bf_offset=args.bf_offset)

    io = PDBIO()
    io.set_structure(structure)
    Path(args.pdb_out).parent.mkdir(parents=True, exist_ok=True)
    io.save(args.pdb_out, select=KeepAll())
    print(f"[OK] Wrote modified PDB -> {args.pdb_out}")

    if args.map_out:
        df = pd.DataFrame(rows, columns=["chain","local_idx","resseq","icode","resname","bfactor","score"])
        Path(args.map_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.map_out, index=False)
        print(f"[OK] Wrote residue map -> {args.map_out}")


if __name__ == "__main__":
    main()

