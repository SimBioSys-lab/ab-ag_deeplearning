#!/usr/bin/env python3
"""
Set PDB B-factors from residue-level predictions for a single PDB.

Inputs
------
- CSV of concatenated residue predictions for multiple PDBs with columns:
  'Chain Type', 'True Label', 'Predicted Label', 'Probability'
- NPZ whose *keys* are exactly the PDB IDs, in the same order as CSV blocks
  (the script maps block i -> NPZ key i; or alphabetical with --order alpha)
- A target PDBID, the input PDB file, and which actual chain IDs correspond to
  antibody light/heavy and antigen chains.

Behavior
--------
- Finds the CSV block for the requested PDBID (by NPZ order).
- Extracts per-chain Probability vectors:
    * 'L...' → antibody light
    * 'H...' → antibody heavy
    * 'AG...' (e.g., 'AGchain_0', 'AGchain_1', ...) → antigen chains
- Assigns those values to residues of the specified PDB chains, in residue order.
- Optionally normalizes the final per-residue vector (percentile/minmax) and
  applies a gamma to boost contrast, then assigns:
    B = bf_offset + bf_scale * score
- Writes modified PDB and optional per-residue CSV map.

Typical usage
-------------
python pdb_preds_to_bfactor.py \
  --csv preds.csv \
  --npz pdb_index.npz \
  --order npz \
  --pdbid 7ZLK \
  --pdb-in 7zlk.pdb \
  --ab-light L \
  --ab-heavy H \
  --ag-chains D \
  --final-norm percentile --final-p-lo 5 --final-p-hi 99 --gamma 1.0 \
  --bf-scale 100 --bf-offset 0 \
  --pdb-out 7zlk_preds.pdb \
  --map-out 7zlk_preds_map.csv
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select


# ---------- I/O helpers ----------

def pdb_ids_from_npz(npz_path: Path, order: str = "npz") -> List[str]:
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.keys())
    if not keys:
        raise ValueError(f"No keys found in NPZ: {npz_path}")
    return [k.upper() for k in (sorted(keys) if order == "alpha" else keys)]


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


def find_pdb_blocks(chain_series: pd.Series) -> List[Tuple[int, int]]:
    """
    A new block starts at any row where 'Chain Type' startswith 'l' (case-insensitive)
    AND (it's the first row OR the previous row does NOT startwith 'l').
    """
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


# ---------- PDB helpers ----------

def get_chain_residues(pdb_path: Path, pdbid: str, chain_ids: List[str]) -> Tuple[Any, Any, Dict[str, list]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdbid, str(pdb_path))
    model = next(structure.get_models())
    chain_map: Dict[str, List] = {}
    for cid in chain_ids:
        if cid not in model:
            raise KeyError(f"Chain '{cid}' not found in PDB.")
        residues = [res for res in model[cid].get_residues() if res.id[0] == " "]  # standard residues only
        chain_map[cid] = residues
    return structure, model, chain_map


class KeepAll(Select):
    def accept_atom(self, atom): return 1


# ---------- Extraction & mapping ----------

def parse_ag_index(tag: str) -> Optional[int]:
    """
    Extract numeric suffix from strings like 'AGchain_0', 'ag_1', etc.
    Returns None if not present.
    """
    s = tag.strip().lower()
    m = re.search(r'ag\w*[_\-]?(\d+)', s)
    if m:
        return int(m.group(1))
    return None


def extract_chain_scores(block_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    From a block (single PDB) return:
      - 'L': probs for L* rows, in row order
      - 'H': probs for H* rows, in row order
      - 'AG': dict index-> probs for each AG group (0,1,2,... if present, else {0: all AG})
    """
    s = block_df["Chain Type"].astype(str).str.lower().str.strip()
    probs = block_df["Probability"].to_numpy(dtype=float)

    out: Dict[str, Any] = {}
    # Light
    mask_l = s.str.startswith("l")
    out["L"] = probs[mask_l]

    # Heavy
    mask_h = s.str.startswith("h")
    out["H"] = probs[mask_h]

    # Antigen groups
    mask_ag = s.str.startswith("ag")
    if mask_ag.any():
        sub = block_df.loc[mask_ag, :]
        st = sub["Chain Type"].astype(str).str.lower().str.strip()
        uniq = st.unique().tolist()
        ag_map: Dict[int, np.ndarray] = {}
        # If there are indexed AG types, split by index; else treat all as AG0
        has_index = any(parse_ag_index(u) is not None for u in uniq)
        if has_index:
            for u in uniq:
                idx = parse_ag_index(u)
                if idx is None:
                    continue
                p = sub.loc[st == u, "Probability"].to_numpy(dtype=float)
                ag_map[idx] = p
            # fill any missing numbers in order they appear
            out["AG"] = {k: ag_map[k] for k in sorted(ag_map)}
        else:
            out["AG"] = {0: sub["Probability"].to_numpy(dtype=float)}
    else:
        out["AG"] = {}

    return out  # {'L': np.array, 'H': np.array, 'AG': {0: np.array, 1: np.array, ...}}


# ---------- Final normalization ----------

def final_normalize(vec: np.ndarray, how: str, p_lo: float, p_hi: float, gamma: float) -> np.ndarray:
    v = vec.astype(np.float32)
    if how == "minmax":
        lo, hi = float(v.min()), float(v.max())
    elif how == "percentile":
        lo, hi = [float(x) for x in np.percentile(v, [p_lo, p_hi])]
    elif how == "none":
        lo, hi = 0.0, 1.0
    else:
        raise ValueError(f"Unknown final-norm: {how}")
    if how != "none":
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            v[:] = 0.0
        else:
            v = (v - lo) / (hi - lo)
            v = np.clip(v, 0.0, 1.0)
    if gamma != 1.0:
        v = np.power(v, float(gamma))
    return v.astype(np.float32)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Write a PDB with B-factors set from residue-level prediction probabilities.")
    ap.add_argument("--csv", required=True, type=Path, help="Predictions CSV")
    ap.add_argument("--npz", required=True, type=Path, help="NPZ whose KEYS are PDB IDs")
    ap.add_argument("--order", choices=["npz", "alpha"], default="npz",
                    help="Order to read NPZ keys (default: npz)")
    ap.add_argument("--pdbid", required=True, help="Target PDB ID (case-insensitive)")
    ap.add_argument("--pdb-in", required=True, type=Path, help="Input PDB file to modify")

    # chain mapping
    ap.add_argument("--ab-light", required=True, help="Actual PDB chain ID for antibody light")
    ap.add_argument("--ab-heavy", required=True, help="Actual PDB chain ID for antibody heavy")
    ap.add_argument("--ag-chains", nargs="*", default=[], help="Actual PDB chain IDs for antigen in order (maps to AGchain_0, AGchain_1, ...)")

    # output & scaling
    ap.add_argument("--final-norm", choices=["none","minmax","percentile"], default="percentile",
                    help="Normalize final per-residue vector before B-factor scaling")
    ap.add_argument("--final-p-lo", type=float, default=5.0)
    ap.add_argument("--final-p-hi", type=float, default=99.0)
    ap.add_argument("--gamma", type=float, default=1.0, help="Power-law contrast after final normalization")
    ap.add_argument("--bf-scale", type=float, default=100.0)
    ap.add_argument("--bf-offset", type=float, default=0.0)
    ap.add_argument("--pdb-out", required=True, type=Path)
    ap.add_argument("--map-out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true", help="Error on length mismatches instead of trimming to min length")
    args = ap.parse_args()

    # Load CSV & NPZ order
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.npz.exists():
        raise FileNotFoundError(f"NPZ not found: {args.npz}")

    df = read_predictions(args.csv)
    blocks = find_pdb_blocks(df["Chain Type"])
    pdb_ids = pdb_ids_from_npz(args.npz, order=args.order)
    print(f"[INFO] Detected {len(blocks)} CSV blocks; NPZ keys={len(pdb_ids)}")

    # Find the requested PDBID
    pid = args.pdbid.upper()
    try:
        idx = pdb_ids.index(pid)
    except ValueError:
        raise SystemExit(f"PDBID '{pid}' not found among NPZ keys: {pdb_ids}")

    if idx >= len(blocks):
        raise SystemExit(f"PDBID index {idx} exceeds number of CSV blocks {len(blocks)}")

    s, e = blocks[idx]
    block_df = df.iloc[s:e+1].copy()

    # Extract per-chain probabilities from the block
    chain_scores = extract_chain_scores(block_df)  # {'L': arr, 'H': arr, 'AG': {i: arr}}
    # Prepare mapping from logical to actual PDB chain IDs
    chain_order_actual: List[str] = [args.ab_light, args.ab_heavy] + list(args.ag_chains)

    # Load PDB and get residue lists in the same order
    structure, model, chain_map = get_chain_residues(args.pdb_in, pid, chain_order_actual)
    print("[INFO] Residues per chain:", {c: len(chain_map[c]) for c in chain_order_actual})

    # Build per-residue scores aligned to actual chains
    per_res_scores: List[float] = []

    # Light
    if chain_scores["L"].size == 0:
        print("[WARN] No L* rows found in CSV block; filling zeros for light chain.")
        l_vec = np.zeros(len(chain_map[args.ab_light]), dtype=np.float32)
    else:
        l_len = len(chain_map[args.ab_light])
        if len(chain_scores["L"]) != l_len:
            msg = (f"[WARN] L length mismatch: CSV={len(chain_scores['L'])} vs PDB residues={l_len}. "
                   f"{'Error (strict)' if args.strict else 'Trimming to min.'}")
            print(msg)
            m = min(len(chain_scores["L"]), l_len)
            if args.strict and m != l_len:
                raise SystemExit("Length mismatch for L chain.")
            l_vec = chain_scores["L"][:m]
            if m < l_len:
                l_vec = np.pad(l_vec, (0, l_len - m), constant_values=0.0)
        else:
            l_vec = chain_scores["L"].astype(np.float32)
    per_res_scores.append(l_vec)

    # Heavy
    if chain_scores["H"].size == 0:
        print("[WARN] No H* rows found in CSV block; filling zeros for heavy chain.")
        h_vec = np.zeros(len(chain_map[args.ab_heavy]), dtype=np.float32)
    else:
        h_len = len(chain_map[args.ab_heavy])
        if len(chain_scores["H"]) != h_len:
            msg = (f"[WARN] H length mismatch: CSV={len(chain_scores['H'])} vs PDB residues={h_len}. "
                   f"{'Error (strict)' if args.strict else 'Trimming to min.'}")
            print(msg)
            m = min(len(chain_scores["H"]), h_len)
            if args.strict and m != h_len:
                raise SystemExit("Length mismatch for H chain.")
            h_vec = chain_scores["H"][:m]
            if m < h_len:
                h_vec = np.pad(h_vec, (0, h_len - m), constant_values=0.0)
        else:
            h_vec = chain_scores["H"].astype(np.float32)
    per_res_scores.append(h_vec)

    # Antigens
    ag_groups = chain_scores["AG"]  # dict index->array
    if len(args.ag_chains) > 0:
        # match AGchain_0 -> first ag chain, etc.
        for i, ag_cid in enumerate(args.ag_chains):
            arr = ag_groups.get(i, None)
            length = len(chain_map[ag_cid])
            if arr is None:
                print(f"[WARN] No AG group index {i} found in CSV; filling zeros for chain {ag_cid}.")
                per_res_scores.append(np.zeros(length, dtype=np.float32))
                continue
            if len(arr) != length:
                msg = (f"[WARN] AG{i} length mismatch: CSV={len(arr)} vs PDB residues={length}. "
                       f"{'Error (strict)' if args.strict else 'Trimming to min.'}")
                print(msg)
                m = min(len(arr), length)
                if args.strict and m != length:
                    raise SystemExit(f"Length mismatch for AG{i} ({ag_cid})")
                vec = arr[:m].astype(np.float32)
                if m < length:
                    vec = np.pad(vec, (0, length - m), constant_values=0.0)
            else:
                vec = arr.astype(np.float32)
            per_res_scores.append(vec)
    else:
        if ag_groups:
            print(f"[WARN] CSV has antigen rows, but --ag-chains was empty; antigen scores dropped.")

    # Concatenate to one vector [L | H | AG0 | AG1 | ...]
    fused = np.concatenate(per_res_scores, axis=0).astype(np.float32)

    # Final normalization + gamma (purely for visualization)
    fused = final_normalize(fused, args.final_norm, args.final_p_lo, args.final_p_hi, args.gamma)

    # Safety
    total_residues = sum(len(chain_map[c]) for c in chain_order_actual)
    if fused.shape[0] != total_residues:
        # This only happens if strict=False and we padded/trimmed
        print(f"[INFO] Adjusted fused length={len(fused)}; total residues={total_residues}")
        fused = fused[:total_residues] if len(fused) > total_residues else np.pad(fused, (0, total_residues - len(fused)), constant_values=0.0)

    # Write B-factors
    offset = 0
    for cid in chain_order_actual:
        reslist = [r for r in model[cid].get_residues() if r.id[0] == " "]
        vals = fused[offset: offset + len(reslist)]
        for res, val in zip(reslist, vals):
            B = float(args.bf_offset + args.bf_scale * float(val))
            for atom in res.get_atoms():
                atom.set_bfactor(B)
        offset += len(reslist)

    # Save PDB
    args.pdb_out.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO(); io.set_structure(structure)
    io.save(str(args.pdb_out), select=KeepAll())
    print(f"[OK] Wrote B-factor-colored PDB → {args.pdb_out}")

    # Optional residue map CSV
    if args.map_out:
        rows = []
        offset = 0
        for cid in chain_order_actual:
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


if __name__ == "__main__":
    main()

