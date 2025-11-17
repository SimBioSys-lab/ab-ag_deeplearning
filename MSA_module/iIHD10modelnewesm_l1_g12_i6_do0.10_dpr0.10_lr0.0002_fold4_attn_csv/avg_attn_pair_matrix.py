import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ ESM token map (your ground truth) ------------------
def id2aa_true_map():
    return {
        0:"CLS", 1:"PAD", 2:"EOS", 3:"UNK",
        4:"L", 5:"A", 6:"G", 7:"V", 8:"S", 9:"E", 10:"R", 11:"T", 12:"I",
        13:"D", 14:"P", 15:"K", 16:"Q", 17:"N", 18:"F", 19:"Y", 20:"M",
        21:"H", 22:"W", 23:"C",
        24:"X", 25:"B", 26:"U", 27:"Z", 28:"O", 29:".", 30:"-"
    }

CANONICAL_ORDER = ["L","A","G","V","S","E","R","T","I","D","P","K","Q","N","F","Y","M","H","W","C"]
AA_TO_IDX = {aa:i for i,aa in enumerate(CANONICAL_ORDER)}

# ------------------ IO helpers ------------------
def load_seq_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}  # each: M x L

def load_head_csv(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = pd.read_csv(path, header=None).values
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{path}: expected square matrix, got {arr.shape}")
    return arr

def load_attn_stack_for_pdb(attn_tpl: str, pdbid: str):
    """
    attn_tpl should contain '{pdb}' placeholder, e.g. '/attn/{pdb}__head*.csv'
    Returns: stack (H,N,N) and file list
    """
    glob_pat = attn_tpl.format(pdb=pdbid)
    files = glob.glob(glob_pat)
    if not files:
        return None, []
    def head_idx(fn: str) -> int:
        m = re.search(r"__head(\d+)\.csv$", fn)
        return int(m.group(1)) if m else 10**9
    files = sorted(files, key=head_idx)
    mats = [load_head_csv(f) for f in files]
    shapes = {M.shape for M in mats}
    if len(shapes) != 1:
        raise ValueError(f"{pdbid}: attention files have mixed shapes: {shapes}")
    stack = np.stack(mats, axis=0)  # (H, N, N)
    return stack, files

# ------------------ normalization (on submatrix only) ------------------
def norm_matrix_sub(M_sub: np.ndarray, how: str, p_lo: float, p_hi: float):
    """Normalize using stats computed on the AB↔AG submatrix only."""
    if how == "none":
        return M_sub.astype(np.float32)
    if how == "minmax":
        lo, hi = float(M_sub.min()), float(M_sub.max())
    elif how == "percentile":
        lo, hi = [float(x) for x in np.percentile(M_sub, [p_lo, p_hi])]
    else:
        raise ValueError(f"Unknown norm: {how}")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(M_sub, dtype=np.float32)
    out = (M_sub - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(
        description="Average fused attention for antibody–antigen residue pairs (AB↔AG-only normalization)."
    )
    ap.add_argument("--seq-npz", required=True, type=Path,
                    help="Sequences .npz; keys=PDB IDs; value=M×L; first row = token IDs (ESM mapping).")
    ap.add_argument("--attn-tpl", required=True, type=str,
                    help="Glob template with {pdb}, e.g. '/path/attn/{pdb}__head*.csv'")
    ap.add_argument("--reduce-heads", choices=["mean","max"], default="mean",
                    help="Fuse heads element-wise.")
    ap.add_argument("--max-after-slice", action="store_true",
                    help="When using --reduce-heads max, compute the max AFTER slicing to AB↔AG per head.")
    ap.add_argument("--sym", action="store_true",
                    help="Use symmetric pair score: 0.5*(A[ab,ag]+A[ag,ab]^T) instead of directional A[ab,ag].")
    ap.add_argument("--attn-norm", choices=["none","minmax","percentile"], default="none",
                    help="Normalization applied on the AB↔AG submatrix ONLY.")
    ap.add_argument("--p-lo", type=float, default=1.0)
    ap.add_argument("--p-hi", type=float, default=99.0)

    ap.add_argument("--matrix-out", required=True, type=Path,
                    help="CSV: 20x20 matrix (rows=antibody AA, cols=antigen AA).")
    ap.add_argument("--pairs-out", required=True, type=Path,
                    help="CSV: long table with (aa_ab, aa_ag, avg_score, count).")
    ap.add_argument("--heatmap-out", type=Path, default=None,
                    help="Optional PNG heatmap.")

    args = ap.parse_args()

    seq_dict = load_seq_npz(args.seq_npz)
    id2aa = id2aa_true_map()

    # accumulators
    sums = np.zeros((20, 20), dtype=np.float64)
    cnts = np.zeros((20, 20), dtype=np.int64)

    used_pdb = 0
    skipped_pdb = 0

    for pdbid, mat in seq_dict.items():
        tokens = mat[0]  # first row, length L
        L = tokens.shape[0]

        # find EOCs (token==24): need at least two
        eocs = np.where(tokens == 24)[0]
        if eocs.size < 2:
            print(f"[WARN] {pdbid}: need ≥2 EOC (24) tokens; found {eocs.size}. Skipped.")
            skipped_pdb += 1
            continue

        # Define antibody / antigen spans (token indices)
        L_end = eocs[0]                  # 0..L_end-1 = L-chain
        H_end = eocs[1]                  # L_end+1..H_end-1 = H-chain
        AB_mask = np.zeros(L, dtype=bool)
        AB_mask[0:L_end] = True
        AB_mask[L_end+1:H_end] = True

        AG_mask = np.zeros(L, dtype=bool)
        AG_mask[H_end+1:L] = True

        # keep only canonical residues (token 4..23)
        is_residue = (tokens >= 4) & (tokens <= 23)
        ab_idx = np.where(AB_mask & is_residue)[0]
        ag_idx = np.where(AG_mask & is_residue)[0]

        if ab_idx.size == 0 or ag_idx.size == 0:
            print(f"[WARN] {pdbid}: no antibody or antigen residues after filtering; skipped.")
            skipped_pdb += 1
            continue

        # load attention for this pdb
        stack, files = load_attn_stack_for_pdb(args.attn_tpl, pdbid)
        if stack is None:
            print(f"[WARN] {pdbid}: no attention head files matched. Skipped.")
            skipped_pdb += 1
            continue

        Hh, N, N2 = stack.shape
        if N != L or N2 != L:
            print(f"[WARN] {pdbid}: attn size {N}x{N2} != token length {L}. Skipped.")
            skipped_pdb += 1
            continue

        # ---- fuse heads (optionally after slicing) ----
        if args.reduce_heads == "mean":
            A = stack.mean(axis=0)  # (L,L)
            # build AB↔AG submatrix
            if args.sym:
                sub = 0.5*(A[np.ix_(ab_idx, ag_idx)] + A[np.ix_(ag_idx, ab_idx)].T)
            else:
                sub = A[np.ix_(ab_idx, ag_idx)]
        else:  # max
            if args.max_after_slice:
                # slice each head first, then take element-wise max across heads
                subs = []
                for h in range(Hh):
                    if args.sym:
                        subs.append(0.5*(stack[h][np.ix_(ab_idx, ag_idx)] +
                                         stack[h][np.ix_(ag_idx, ab_idx)].T))
                    else:
                        subs.append(stack[h][np.ix_(ab_idx, ag_idx)])
                sub = np.max(np.stack(subs, axis=0), axis=0)
            else:
                # element-wise max on full matrix, then slice (equivalent numerically)
                A = stack.max(axis=0)
                if args.sym:
                    sub = 0.5*(A[np.ix_(ab_idx, ag_idx)] + A[np.ix_(ag_idx, ab_idx)].T)
                else:
                    sub = A[np.ix_(ab_idx, ag_idx)]

        # ---- normalize ONLY using the AB↔AG submatrix ----
        sub = norm_matrix_sub(sub, args.attn_norm, args.p_lo, args.p_hi)

        # ---- accumulate by AA type pairs ----
        aa1 = np.array([id2aa.get(int(t), "UNK") for t in tokens], dtype=object)
        aa_ab = aa1[ab_idx]
        aa_ag = aa1[ag_idx]

        ab_bins = np.vectorize(AA_TO_IDX.get)(aa_ab)
        ag_bins = np.vectorize(AA_TO_IDX.get)(aa_ag)

        for i, bi in enumerate(ab_bins):
            for j, bj in enumerate(ag_bins):
                if bi is None or bj is None:
                    continue
                sums[bi, bj] += float(sub[i, j])
                cnts[bi, bj] += 1

        used_pdb += 1

    if used_pdb == 0:
        raise SystemExit("No PDBs processed successfully. Check EOC tokens and attention file template.")

    # compute averages
    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.where(cnts > 0, sums / cnts, np.nan)

    # write matrix CSV (20x20 with headers)
    matrix_df = pd.DataFrame(avg, index=CANONICAL_ORDER, columns=CANONICAL_ORDER)
    args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(args.matrix_out)
    print(f"[OK] Wrote 20x20 matrix → {args.matrix_out}")

    # long table
    rows = []
    for i, aa_i in enumerate(CANONICAL_ORDER):
        for j, aa_j in enumerate(CANONICAL_ORDER):
            rows.append({"aa_ab": aa_i, "aa_ag": aa_j,
                         "avg_score": None if np.isnan(avg[i,j]) else float(avg[i,j]),
                         "count": int(cnts[i,j])})
    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(args.pairs_out, index=False)
    print(f"[OK] Wrote pair table → {args.pairs_out}")

    # optional heatmap
    if args.heatmap_out:
        plt.figure(figsize=(7,6))
        im = plt.imshow(avg, interpolation="nearest", aspect="auto")
        plt.xticks(range(20), CANONICAL_ORDER, rotation=90)
        plt.yticks(range(20), CANONICAL_ORDER)
        plt.xlabel("Antigen residue (AA)")
        plt.ylabel("Antibody residue (AA)")
        plt.title(f"Avg attention {'sym' if args.sym else 'AB→AG'} over {used_pdb} PDBs")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        args.heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.heatmap_out, dpi=200)
        plt.close()
        print(f"[OK] Wrote heatmap → {args.heatmap_out}")

    print(f"[STATS] Used PDBs: {used_pdb}, Skipped: {skipped_pdb}")

if __name__ == "__main__":
    main()

