#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, csv, itertools, gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             auc, precision_score, recall_score)

from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel

# ─────────────────────────── 0.  runtime / device ───────────────────────────
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── helper: clean state-dict ──────────────────────
def load_state_cpu(path):
    sd = torch.load(path, map_location="cpu")
    sd.pop("n_averaged", None)                              # drop SWA counter
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

# ─────────────────────────── 1.  read checkpoint list ──────────────────────
with open("new45_iPPIPara_files") as f:
    ckpt_paths = [ln.strip() for ln in f if ln.strip()]
if not ckpt_paths:
    raise RuntimeError("new45_iPPIPara_files is empty!")

# ─────────────────────────── 2.  dataset & loader ──────────────────────────
test_cfg = dict(
    batch_size   = 1,
    sequence_npz = "cleaned_para_test_sequences_1600.npz",
    data_npz     = "cleaned_para_test_interfaces_1600.npz",
    edge_npz     = "cleaned_para_test_edges_1600.npz",
    max_len      = 1600,
    vocab_size   = 23,
    num_classes  = 2,
)

def collate_fn(batch):
    seq, pt, edge = zip(*batch)
    seq = torch.stack(seq)
    pt  = torch.tensor(np.array(pt), dtype=torch.long)
    mE  = max(e.shape[0] for e in edge)
    pads = []
    for e in edge:
        pad = -torch.ones((2, mE), dtype=torch.long)
        pad[:, :e.shape[0]] = e.T.clone()
        pads.append(pad)
    return torch.stack(pads), seq, pt

test_set = SequenceParatopeDataset(
    data_file     = test_cfg["data_npz"],
    sequence_file = test_cfg["sequence_npz"],
    edge_file     = test_cfg["edge_npz"],
    max_len       = test_cfg["max_len"],
)
test_loader = DataLoader(test_set, batch_size=test_cfg["batch_size"],
                         shuffle=False, collate_fn=collate_fn)

# ─────────────────────────── 3.  chain-aware evaluator ─────────────────────
def evaluate_model(model, name):
    metrics = {
        "Lchain":  {"true": [], "pred": [], "prob": []},
        "Hchain":  {"true": [], "pred": [], "prob": []},
        "Antibody":{"true": [], "pred": [], "prob": []},
        "Antigen": {"true": [], "pred": [], "prob": []},
    }
    with open(f"{name}_predictions.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["Chain Type", "True", "Pred", "Prob"])

        with torch.no_grad():
            for pads, seq, pt in test_loader:
                pads, seq, pt = pads.to(device), seq.to(device), pt.to(device)
                with autocast("cuda"):
                    logits, _ = model(sequences=seq, padded_edges=pads,
                                      return_attention=True)
                logits = logits.squeeze(1)                 # (B,L,2) or (B,2)
                prob = torch.softmax(logits, -1)
                pred = prob.argmax(-1)

                eoc = np.where(seq[0,0].cpu().numpy() == 22)[0]
                ctype = ["Lchain","Hchain"]+[f"AGchain_{i}" for i in range(len(eoc)-2)]
                split = [0]+eoc.tolist()

                for ct in ctype[2:]:
                    metrics.setdefault(ct, {"true": [], "pred": [], "prob": []})

                for i, ct in enumerate(ctype):
                    s, e = split[i], split[i+1]
                    msk  = pt[0,s:e] >= 0
                    if msk.sum()==0: continue
                    t  = pt[0,s:e][msk].cpu().numpy()
                    p  = pred[0,s:e][msk].cpu().numpy()
                    pr = prob[0,s:e,1][msk].cpu().numpy()

                    for k,v in zip(("true","pred","prob"),(t,p,pr)):
                        metrics[ct][k].extend(v)

                    agg = "Antibody" if ct in {"Lchain","Hchain"} else "Antigen"
                    for k,v in zip(("true","pred","prob"),(t,p,pr)):
                        metrics[agg][k].extend(v)

                    for t1,p1,pr1 in zip(t,p,pr):
                        writer.writerow([ct,t1,p1,pr1])

    # print Antibody aggregate
    t = np.array(metrics["Antibody"]["true"])
    p = np.array(metrics["Antibody"]["pred"])
    pr= np.array(metrics["Antibody"]["prob"])
    acc  = (t==p).mean()
    prec = precision_score(t,p,zero_division=0)
    rec  = recall_score(t,p,zero_division=0)
    roc  = roc_auc_score(t,pr) if len(np.unique(t))==2 else float("nan")
    pr_p,pr_r,_ = precision_recall_curve(t,pr)
    pr_auc = auc(pr_r,pr_p)
    print(f"[{name}] Antibody: Acc {acc:.3f}  P {prec:.3f}  "
          f"R {rec:.3f}  ROC {roc:.3f}  PR {pr_auc:.3f}")

# ─────────────────────────── 4.  evaluate original checkpoints ────────────
for path in ckpt_paths:
    print(f"\nTesting checkpoint: {path}")
    parts = path.split("_")
    n_l   = int(parts[1][1:]); n_g = int(parts[2][1:]); n_i = int(parts[3][1:])
    dp    = float(parts[4][2:]); do = float(parts[5][3:])

    model = ClassificationModel(
        vocab_size=test_cfg["vocab_size"], seq_len=test_cfg["max_len"],
        embed_dim=256, num_heads=16, dropout=do, drop_path_rate=dp,
        num_layers=n_l, num_gnn_layers=n_g, num_int_layers=n_i,
        num_classes=test_cfg["num_classes"],
    ).to(device)

    model.load_state_dict(load_state_cpu(path), strict=False)
    model.eval()
    evaluate_model(model, os.path.basename(path))
    del model; torch.cuda.empty_cache(); gc.collect()

# ─────────────────────────── 5.  PCA over checkpoints ─────────────────────
print("\n► PCA over checkpoints …")
state0 = load_state_cpu(ckpt_paths[0])
key_order   = list(state0.keys())
shape_order = [state0[k].shape for k in key_order]
dtype_order = [state0[k].dtype for k in key_order]

def flat_state(path):
    sd = load_state_cpu(path)
    return torch.cat([sd[k].flatten() for k in key_order]).float().numpy()

matrix = np.stack([flat_state(p) for p in ckpt_paths])
pca = PCA(n_components=2, random_state=0).fit(matrix)
mu, pc1, pc2 = pca.mean_.astype(np.float32), *pca.components_.astype(np.float32)
print("   Explained variance ratio:", pca.explained_variance_ratio_)

# save projection of each checkpoint
proj = pca.transform(matrix)                             # (N,2)
with open("checkpoint_pc_projection.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["checkpoint","PC1","PC2"])
    for path,(x,y) in zip(ckpt_paths, proj):
        w.writerow([os.path.basename(path), f"{x:.6f}", f"{y:.6f}"])
print("   → wrote per-checkpoint PC coordinates to checkpoint_pc_projection.csv")

def vec_to_state(vec):
    sd, i = {}, 0
    for k, shp, dt in zip(key_order, shape_order, dtype_order):
        n = int(np.prod(shp))
        sd[k] = torch.tensor(vec[i:i+n], dtype=dt).view(*shp)
        i += n
    return sd

# ─────────────────────────── 6.  PCA grid sweep ───────────────────────────
GRID = np.linspace(-3, 3, 9)
for a, b in itertools.product(GRID, GRID):
    name = f"PCA_a{a:+.1f}_b{b:+.1f}"
    vec  = mu + a*pc1 + b*pc2

    model = ClassificationModel(
        vocab_size=test_cfg["vocab_size"], seq_len=test_cfg["max_len"],
        embed_dim=256, num_heads=16, dropout=do, drop_path_rate=dp,
        num_layers=n_l, num_gnn_layers=n_g, num_int_layers=n_i,
        num_classes=test_cfg["num_classes"],
    ).to(device)

    model.load_state_dict(vec_to_state(vec), strict=False)
    model.eval()
    print(f"\nEvaluating {name} …")
    evaluate_model(model, name)
    del model; torch.cuda.empty_cache(); gc.collect()

