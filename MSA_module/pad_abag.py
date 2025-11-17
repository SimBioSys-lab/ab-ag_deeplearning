#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Separate antibody and antigen parts in sequences / labels / edges.

Usage
-----
python split_ab_ag_full.py \
       --seq  cleaned_para_test_sequences_1600.npz \
       --lab  cleaned_para_test_interfaces_1600.npz \
       --edge cleaned_para_test_edges_1600.npz
"""

import os, argparse, numpy as np

EOC_TOKEN      = 24     # end-of-chain token in sequences
SEQ_PAD_VALUE  = 1      # pad for sequence rows
LAB_PAD_VALUE  = -1     # pad for labels
SEQ_LENGTH     = 1600   # expected total length

# ───────────────────────── helpers ─────────────────────────
def find_second_eoc(row: np.ndarray) -> int:
    """Return index of the 2nd EOC; if <2 tokens, use last position."""
    idx = np.where(row == EOC_TOKEN)[0]
    return idx[1] if len(idx) >= 2 else len(row) - 1


def split_sequence(row: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad antibody segment in place, shift antigen to front."""
    ab = np.full_like(row, SEQ_PAD_VALUE)
    ag = np.full_like(row, SEQ_PAD_VALUE)

    ab[:offset] = row[:offset]            # antibody
    ag_len      = len(row) - offset
    ag[:ag_len] = row[offset:]            # antigen (shifted)

    return ab, ag


def split_labels(labels: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Handle 1-D or 2-D label arrays."""
    if labels.ndim == 1:
        labels = labels[None, :]

    ab_rows, ag_rows = [], []
    for r in labels:
        ab = np.full_like(r, LAB_PAD_VALUE)
        ag = np.full_like(r, LAB_PAD_VALUE)

        ab[:offset] = r[:offset]
        ag_len      = len(r) - offset
        ag[:ag_len] = r[offset:]

        ab_rows.append(ab)
        ag_rows.append(ag)

    ab_arr = np.stack(ab_rows) if len(ab_rows) > 1 else ab_rows[0]
    ag_arr = np.stack(ag_rows) if len(ag_rows) > 1 else ag_rows[0]
    return ab_arr, ag_arr


def split_edges(edges: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep edges strictly within each part; renumber antigen indices."""
    if edges.size == 0:
        return edges.copy(), edges.copy()

    ab_mask = (edges[:, 0] < offset) & (edges[:, 1] < offset)
    ag_mask = (edges[:, 0] >= offset) & (edges[:, 1] >= offset)

    ab_edges = edges[ab_mask].copy()

    ag_edges = edges[ag_mask].copy()
    ag_edges -= offset                      # renumber for antigen

    return ab_edges, ag_edges
# ───────────────────────── main ───────────────────────────
def main(seq_npz, lab_npz, edge_npz):
    base = os.path.splitext(seq_npz)[0]

    seqs   = np.load(seq_npz,  allow_pickle=True)
    labels = np.load(lab_npz,  allow_pickle=True)
    edges  = np.load(edge_npz, allow_pickle=True)

    ab_seq_dict, ag_seq_dict   = {}, {}
    ab_lab_dict, ag_lab_dict   = {}, {}
    ab_edge_dict, ag_edge_dict = {}, {}

    for key in seqs.files:
        seq_arr   = seqs[key]          # (64,1600)
        label_arr = labels[key]        # (64,1600) or (1600,)
        edge_arr  = edges[key]         # (E,2)

        # locate 2nd EOC once per sample
        offset = find_second_eoc(seq_arr[0]) + 1  # +1: first antigen idx

        # split sequences
        ab_rows, ag_rows = [], []
        for row in seq_arr:
            ab_r, ag_r = split_sequence(row, offset)
            ab_rows.append(ab_r)
            ag_rows.append(ag_r)
        ab_seq_dict[key] = np.stack(ab_rows)
        ag_seq_dict[key] = np.stack(ag_rows)

        # split labels
        ab_lab, ag_lab = split_labels(label_arr, offset)
        ab_lab_dict[key] = ab_lab
        ag_lab_dict[key] = ag_lab

        # split edges
        ab_e, ag_e = split_edges(edge_arr, offset)
        ab_edge_dict[key] = ab_e
        ag_edge_dict[key] = ag_e

        print(f"processed '{key}': offset={offset}, "
              f"ab_edges={len(ab_e)}, ag_edges={len(ag_e)}")

    # save
    np.savez_compressed(f"{base}_ab_seqs.npz",   **ab_seq_dict)
    np.savez_compressed(f"{base}_ag_seqs.npz",   **ag_seq_dict)
    np.savez_compressed(f"{base}_ab_labels.npz", **ab_lab_dict)
    np.savez_compressed(f"{base}_ag_labels.npz", **ag_lab_dict)
    np.savez_compressed(f"{base}_ab_edges.npz",  **ab_edge_dict)
    np.savez_compressed(f"{base}_ag_edges.npz",  **ag_edge_dict)

    print("\nFinished ✅")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split antibody / antigen data.")
    p.add_argument("--seq",  required=True, help="sequences .npz")
    p.add_argument("--lab",  required=True, help="labels    .npz")
    p.add_argument("--edge", required=True, help="edges     .npz")
    args = p.parse_args()
    main(args.seq, args.lab, args.edge)

