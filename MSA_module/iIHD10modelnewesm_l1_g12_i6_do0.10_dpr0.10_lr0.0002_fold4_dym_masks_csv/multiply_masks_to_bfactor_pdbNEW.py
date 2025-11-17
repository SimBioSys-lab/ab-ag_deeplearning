#!/usr/bin/env python3
"""
Label-free fusion of DyM masks → single PDB with B-factors = fused score.

- No ground-truth labels required.
- Per-mask normalization over actual residues only (padding/EOC ignored), optional per-chain.
- Fusion options: topk_mean (default), mean, geomean, max, quantile.
- Final across-residue normalization so visualization is informative.

Example:
  python multiply_masks_to_bfactor_pdb.py \
    --pdbid 7ZLK \
    --pdb-in 7zlk.pdb \
    --chains L H D \
    --eoc 1 \
    --mask-dir /path/to/masks \
    --fusion topk_mean --k-top 5 \
    --norm percentile --p-lo 5 --p-hi 99 --norm-per-chain \
    --final-norm percentile --final-p-lo 5 --final-p-hi 99 --gamma 1.0 \
    --bf-scale 100 --bf-offset 0 \
    --pdb-out 7zlk_fused.pdb \
    --map-out 7zlk_fused_map.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.NeighborSearch import NeighborSearch  # (not used by default here, but harmless if kept)

# ------------------ DyM mask loading ------------------

def load_mask_csv(path: Path) -> np.ndarray:
    """
    Load a DyM mask CSV into a 1D float32 vector.
    Supported:
      - 'mask' column (any case): token_idx,mask
      - 2 columns: 'token_idx' + value column
      - 1 column: values only
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "mask" in cols:
        arr = df[cols["mask"]].to_numpy(dtype=np.float32)
        return arr.reshape(-1)

    if df.shape[1] == 2 and "token_idx" in cols:
        other = [c for c in df.columns if c.lower() != "token_idx"][0]
        arr = df[other].to_numpy(dtype=np.float32)
        return arr.reshape(-1)

    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)

    raise ValueError(f"{path.name}: cannot infer mask column from {list(df.columns)}")

# ------------------ PDB & mapping ------------------

def get_chain_residues(pdb_path: Path, pdbid: str, chain_ids: List[str]):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdbid, str(pdb_path))
    model = next(structure.get_models())
    chain_map: Dict[str, List] = {}
    for cid in chain_ids:
        if cid not in model:
            raise KeyError(f"Chain '{cid}' not found in PDB.")
        residues = [res for res in model[cid].get_residues() if res.id[0] == " "]
        chain_map[cid] = residues
    return structure, model, chain_map

def build_token_layout(s_len: int, chain_map: Dict[str, List], chain_order: List[str], eoc: int):
    """Return (used_mask: bool[s_len], spans: [(start, L, cid), ...])."""
    used = np.zeros(s_len, dtype=bool)
    spans: List[Tuple[int,int,str]] = []
    idx = 0
    for k, cid in enumerate(chain_order):
        L = len(chain_map[cid])
        if idx + L > s_len:
            raise ValueError(
                f"Mask length {s_len} too short; need {idx+L} to cover chain {cid} ({L} residues)."
            )
        used[idx:idx+L] = True
        spans.append((idx, L, cid))
        idx += L
        if k < len(chain_order) - 1:
            idx += eoc
    return used, spans

def extract_residue_vector_from_tokens(s: np.ndarray, chain_map: Dict[str, List],
                                       chain_order: List[str], eoc: int) -> np.ndarray:
    """Concatenate per-chain segments from the token vector into a per-residue vector."""
    vals = []
    idx = 0
    for k, cid in enumerate(chain_order):
        L = len(chain_map[cid])
        seg = s[idx:idx+L]
        if len(seg) < L:
            raise ValueError(f"Insufficient tokens while assigning chain {cid}")
        vals.append(seg)
        idx += L
        if k < len(chain_order) - 1:
            idx += eoc
    return np.concatenate(vals, axis=0).astype(np.float32)

# ------------------ Normalization helpers ------------------

def norm_basic(x: np.ndarray, how: str, p_lo: float, p_hi: float) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    if how == "none":
        return x.astype(np.float32)
    if how == "minmax":
        lo, hi = float(np.min(x)), float(np.max(x))
    elif how == "percentile":
        lo, hi = [float(v) for v in np.percentile(x, [p_lo, p_hi])]
    else:
        raise ValueError(f"Unknown norm: {how}")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def _norm_within_mask(seg: np.ndarray, seg_mask: np.ndarray, how: str, p_lo: float, p_hi: float) -> np.ndarray:
    """Normalize only positions where seg_mask=True; leave others' relative scale."""
    out = seg.copy()
    vals = seg[seg_mask]
    vals_n = norm_basic(vals, how, p_lo, p_hi)
    out[seg_mask] = vals_n
    return out

def normalize_mask_on_actual_tokens(s: np.ndarray, used_mask: np.ndarray, spans: List[Tuple[int,int,str]],
                                    how: str, p_lo: float, p_hi: float, per_chain: bool,
                                    domain_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize only real-residue tokens; optionally per chain.
    If domain_mask is provided (same length as s), normalization stats are computed on
    used_mask & domain_mask jointly (values outside keep their relative scale).
    """
    s = s.astype(np.float32, copy=True)
    if how == "none":
        return s
    base_mask = used_mask.copy()
    if domain_mask is not None:
        base_mask &= domain_mask

    if per_chain:
        for start, L, _cid in spans:
            m = np.zeros_like(s, dtype=bool)
            m[start:start+L] = True
            if domain_mask is not None:
                m &= domain_mask
            seg = s[start:start+L]
            seg_mask = m[start:start+L]
            if seg_mask.any():
                s[start:start+L] = _norm_within_mask(seg, seg_mask, how, p_lo, p_hi)
            else:
                s[start:start+L] = norm_basic(seg, how, p_lo, p_hi)  # fallback
    else:
        if base_mask.any():
            vals = s[base_mask]
            s[base_mask] = norm_basic(vals, how, p_lo, p_hi)
        else:
            s[used_mask] = norm_basic(s[used_mask], how, p_lo, p_hi)
    return s

def final_normalize_residue_scores(vec: np.ndarray, how: str, p_lo: float, p_hi: float, gamma: float) -> np.ndarray:
    v = norm_basic(vec, how, p_lo, p_hi)
    if gamma != 1.0:
        v = np.power(v, float(gamma))
    return v.astype(np.float32)

# ------------------ Exact mask sequence ------------------

def build_exact_sequence(mask_dir: Path, pdbid: str) -> List[Path]:
    """
    Return ordered Paths following the exact sequence you provided earlier.
    Missing files are skipped with a warning.
    """
    pid = pdbid.lower()
    def first(glob_pat: str) -> Optional[Path]:
        hits = sorted(Path(mask_dir).glob(glob_pat))
        return hits[0] if hits else None

    seq: List[Path] = []
    def add(p: Optional[Path], label: str):
        if p is None: print(f"[WARN] missing {label}")
        else: seq.append(p)

    add(first(f"mc_model.cg_model.core_model.attn_dym.0__{pid}.csv"), "core_model.attn_dym.0")
    add(first(f"mc_model.cg_model.core_model.ffn_dym.0__{pid}.csv"),  "core_model.ffn_dym.0")
    add(first(f"mc_model.cg_model.fc_dym__{pid}.csv"),                 "cg_model.fc_dym")
    for i in range(12):
        add(first(f"mc_model.cg_model.gnn_dym.{i}__{pid}.csv"),        f"cg_model.gnn_dym.{i}")
        add(first(f"mc_model.cg_model.ffn_dym.{i}__{pid}.csv"),        f"cg_model.ffn_dym.{i}")
    for i in range(6):
        add(first(f"mc_model.row_dym.{i}__{pid}.csv"),                 f"row_dym.{i}")
        add(first(f"mc_model.ffn_dym.{i}__{pid}.csv"),                 f"ffn_dym.{i}")

    if not seq:
        raise SystemExit("No masks found that match the exact sequence.")
    return seq

# ------------------ Fusion methods ------------------

def fuse_topk_mean(mats: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, mats.shape[0])))
    part = np.partition(mats, -k, axis=0)[-k:]
    return part.mean(axis=0)

def fuse_geomean(mats: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(np.mean(np.log(np.clip(mats, eps, 1.0)), axis=0))

def fuse_mean(mats: np.ndarray) -> np.ndarray:
    return mats.mean(axis=0)

def fuse_max(mats: np.ndarray) -> np.ndarray:
    return mats.max(axis=0)

def fuse_quantile(mats: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(mats, q, axis=0)

# ------------------ Main ------------------

class KeepAll(Select):
    def accept_atom(self, atom): return 1

def main():
    ap = argparse.ArgumentParser(description="Label-free fusion of DyM masks → single B-factor-colored PDB.")
    # structure & masks
    ap.add_argument("--pdbid", required=True)
    ap.add_argument("--pdb-in", required=True, type=Path)
    ap.add_argument("--chains", nargs="+", required=True, help="Token chain order, e.g. L H D")
    ap.add_argument("--eoc", type=int, default=1, help="EOC tokens between chains (default 1)")
    ap.add_argument("--mask-dir", required=True, type=Path)

    # per-mask normalization
    ap.add_argument("--norm", choices=["none","minmax","percentile"], default="percentile",
                    help="Per-mask normalization on actual residues")
    ap.add_argument("--p-lo", type=float, default=5.0)
    ap.add_argument("--p-hi", type=float, default=99.0)
    ap.add_argument("--norm-per-chain", action="store_true")

    # fusion
    ap.add_argument("--fusion", choices=["topk_mean","geomean","mean","max","quantile"], default="topk_mean")
    ap.add_argument("--k-top", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--eps", type=float, default=1e-6)

    # final normalization & output
    ap.add_argument("--final-norm", choices=["none","minmax","percentile"], default="percentile")
    ap.add_argument("--final-p-lo", type=float, default=5.0)
    ap.add_argument("--final-p-hi", type=float, default=99.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--bf-scale", type=float, default=100.0)
    ap.add_argument("--bf-offset", type=float, default=0.0)
    ap.add_argument("--pdb-out", required=True, type=Path)
    ap.add_argument("--map-out", type=Path, default=None)
    args = ap.parse_args()

    # Structure & layout
    structure, model, chain_map = get_chain_residues(args.pdb_in, args.pdbid, args.chains)
    total_res = sum(len(chain_map[c]) for c in args.chains)
    print("[INFO] Residues per chain:", {c: len(chain_map[c]) for c in args.chains})
    print(f"[INFO] Total residues: {total_res}")

    # Discover masks (exact sequence)
    mask_paths = build_exact_sequence(args.mask_dir, args.pdbid)
    print(f"[INFO] Found {len(mask_paths)} mask files.")

    # Collect normalized per-residue masks
    mats = []
    for order, path in enumerate(mask_paths):
        s = load_mask_csv(path)
        used_mask, spans = build_token_layout(len(s), chain_map, args.chains, args.eoc)
        s_norm = normalize_mask_on_actual_tokens(
            s, used_mask, spans,
            args.norm, args.p_lo, args.p_hi,
            args.norm_per_chain,
            domain_mask=None
        )
        m = extract_residue_vector_from_tokens(s_norm, chain_map, args.chains, args.eoc)
        mats.append(m)

    if len(mats) == 0:
        raise SystemExit("No masks to fuse (none found).")

    M = np.stack(mats, axis=0)  # [K, N]
    print(f"[INFO] Fusing {M.shape[0]} stages.")

    # Fusion
    if args.fusion == "topk_mean":
        fused = fuse_topk_mean(M, args.k_top)
    elif args.fusion == "geomean":
        fused = fuse_geomean(M, args.eps)
    elif args.fusion == "mean":
        fused = fuse_mean(M)
    elif args.fusion == "max":
        fused = fuse_max(M)
    else:
        fused = fuse_quantile(M, args.quantile)

    # Final across-residue scaling + gamma
    fused = final_normalize_residue_scores(fused, args.final_norm, args.final_p_lo, args.final_p_hi, args.gamma)

    # Safety check
    assert fused.shape[0] == total_res, "Fused vector length doesn't match total residues."

    # Write B-factors
    offset = 0
    for cid in args.chains:
        reslist = [r for r in model[cid].get_residues() if r.id[0] == " "]
        vals = fused[offset: offset + len(reslist)]
        for res, val in zip(reslist, vals):
            B = float(args.bf_offset + args.bf_scale * float(val))
            for atom in res.get_atoms():
                atom.set_bfactor(B)
        offset += len(reslist)

    args.pdb_out.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO(); io.set_structure(structure)
    io.save(str(args.pdb_out), select=KeepAll())
    print(f"[OK] Wrote B-factor-colored PDB → {args.pdb_out}")

    # Residue map (safe offset logic; no index reuse bug)
    if args.map_out:
        rows = []
        offset = 0
        for cid in args.chains:
            reslist = [r for r in model[cid].get_residues() if r.id[0] == " "]
            vals = fused[offset: offset + len(reslist)]
            for j, (res, val) in enumerate(zip(reslist, vals)):
                resseq = int(res.id[1])
                icode = (res.id[2] if isinstance(res.id[2], str) else "").strip()
                rows.append({
                    "chain": cid,
                    "local_idx": offset + j,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": res.get_resname(),
                    "score": float(val),
                    "bfactor": float(args.bf_offset + args.bf_scale * float(val)),
                })
            offset += len(reslist)
        pd.DataFrame(rows).to_csv(args.map_out, index=False)
        print(f"[OK] Wrote residue map → {args.map_out}")

class KeepAll(Select):
    def accept_atom(self, atom): return 1

if __name__ == "__main__":
    main()

