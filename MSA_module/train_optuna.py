#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna search (architecture only): tune num_gnn_layers and num_int_layers.
Includes a worker-safe NPZ dataset so multiple DataLoader workers won't corrupt np.load handles.

Run:
  python train_optuna_fixed.py --trials 40 --proxy-epochs 18
"""

import argparse, math, warnings, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# Quiet a noisy timm deprecation (harmless)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

# --- Your model import (unchanged) ---
from Models_fullnew import ClassificationModel

# ---------------- CONFIG (fixed) ----------------
CONFIG = {
    # data files
    'sequence_file':  'MIPE_tv_esmsequences_1600.npz',
    'data_file':      'MIPE_tv_esminterfaces_1600.npz',
    'edge_file':      'MIPE_tv_esmedges_1600.npz',

    # model base (kept fixed)
    'vocab_size':     31,
    'seq_len':        1600,
    'embed_dim':      256,
    'num_heads':      16,
    'dropout':        0.10,
    'num_layers':     1,
    'drop_path_rate': 0.10,
    'num_classes':    2,

    # optimization (kept fixed)
    'batch_size':     4,
    'accum_steps':    2,
    'learning_rate':  2e-4,
    'weight_decay':   1e-2,
    'warmup_epochs':  10,
    'num_epochs':     50,
    'swa_start':      20,
    'max_grad_norm':  0.10,

    # pos-weight anneal (kept fixed)
    'weight_start':   10.0,
    'weight_end':     1.0,
    'weight_anneal_epochs': 20,

    # data loading
    'train_fraction': 0.7,
    'num_workers':    4,
    'pin_memory':     True,
    'persistent_workers': True,
}

# ---------------- Worker-safe Dataset ----------------
class SafeSequenceParatopeDataset(Dataset):
    """
    Opens NPZ archives lazily per worker process to avoid sharing ZipFile handles across forks.
    Yields: (sequence, label, edges) like your original dataset.
    """
    def __init__(self, sequence_file, data_file, edge_file, max_len):
        self.sequence_file = sequence_file
        self.data_file = data_file
        self.edge_file = edge_file
        self.max_len = max_len

        # Determine aligned keys once (in main process) without keeping handles open
        with np.load(sequence_file, allow_pickle=False) as s, \
             np.load(data_file, allow_pickle=False) as d, \
             np.load(edge_file, allow_pickle=False) as e:
            self.keys = sorted(set(s.files) & set(d.files) & set(e.files))

        # Worker-local NPZ handles (created lazily in each worker)
        self._seq = None
        self._dat = None
        self._edg = None

    def _ensure_open(self):
        # Open handles if not already open in this process / worker
        if self._seq is None:
            self._seq = np.load(self.sequence_file, allow_pickle=False, mmap_mode=None)
        if self._dat is None:
            self._dat = np.load(self.data_file, allow_pickle=False, mmap_mode=None)
        if self._edg is None:
            self._edg = np.load(self.edge_file, allow_pickle=False, mmap_mode=None)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._ensure_open()
        k = self.keys[idx]
        seq   = self._seq[k]     # expected shape: (L,) or (L,feat)
        label = self._dat[k]     # expected int labels per token or one label
        edges = self._edg[k]     # expected shape: (E, 2) or similar
        return torch.from_numpy(seq), label, torch.from_numpy(edges)

    def __getstate__(self):
        # Avoid pickling NPZ handles when forking
        state = self.__dict__.copy()
        state['_seq'] = None
        state['_dat'] = None
        state['_edg'] = None
        return state

# ---------------- Collate (same logic you use) ----------------
def custom_collate_fn(batch):
    seqs, labels, edges = zip(*batch)
    seqs   = torch.stack(seqs)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    max_e  = max(e.shape[0] for e in edges)
    pads   = []
    for e in edges:
        pad = -torch.ones((2, max_e), dtype=torch.long)
        pad[:, :e.shape[0]] = e.T.clone().detach()
        pads.append(pad)
    edges = torch.stack(pads)
    return edges, seqs, labels

# ---------------- Training / Eval ----------------
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running = 0.0
    for i, (e, s, l) in enumerate(loader):
        e, s, l = e.to(device), s.to(device), l.to(device)
        if i % CONFIG['accum_steps'] == 0:
            optimizer.zero_grad(set_to_none=True)
        with autocast():
            out, _ = model(sequences=s, padded_edges=e, return_attention=True)
            loss = criterion(out.view(-1, 2), l.view(-1)) / CONFIG['accum_steps']
        scaler.scale(loss).backward()
        if (i + 1) % CONFIG['accum_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        running += loss.item() * CONFIG['accum_steps']
    return running / len(loader)

@torch.no_grad()
def eval_aupr(model, loader):
    model.eval()
    ys, ps = [], []
    for e, s, l in loader:
        e, s, l = e.to(device), s.to(device), l.to(device)
        logits = model(sequences=s, padded_edges=e)
        prob = torch.softmax(logits, dim=-1)[..., 1]
        ys.append(l.view(-1).cpu()); ps.append(prob.view(-1).float().cpu())
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    mask = (y != -1)
    if mask.sum() == 0:
        return 0.0
    return float(average_precision_score(y[mask], p[mask]))

def build_loaders(dataset):
    N = len(dataset)
    idx = np.arange(N)
    tr_idx, vl_idx = train_test_split(
        idx, test_size=(1.0 - CONFIG['train_fraction']), random_state=42, shuffle=True
    )
    tr_loader = DataLoader(
        Subset(dataset, tr_idx),
        batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers'],
        collate_fn=custom_collate_fn
    )
    vl_loader = DataLoader(
        Subset(dataset, vl_idx),
        batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers'],
        collate_fn=custom_collate_fn
    )
    return tr_loader, vl_loader

def make_model(num_gnn_layers, num_int_layers):
    model = ClassificationModel(
        vocab_size=CONFIG['vocab_size'], seq_len=CONFIG['seq_len'],
        embed_dim=CONFIG['embed_dim'], num_heads=CONFIG['num_heads'],
        dropout=CONFIG['dropout'], num_layers=CONFIG['num_layers'],
        num_gnn_layers=num_gnn_layers, num_int_layers=num_int_layers,
        num_classes=CONFIG['num_classes'], drop_path_rate=CONFIG['drop_path_rate']
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(device)

# ---------------- Optuna objective ----------------
import optuna
from optuna.pruners import SuccessiveHalvingPruner

def objective(trial, dataset, proxy_epochs):
    # Only tune these two:
    num_gnn_layers = trial.suggest_categorical("num_gnn_layers", [2, 4, 6, 8, 10, 12])
    num_int_layers = trial.suggest_categorical("num_int_layers", [2, 4, 6, 8])

    tr_loader, vl_loader = build_loaders(dataset)

    model = make_model(num_gnn_layers, num_int_layers)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler()

    def lr_lambda(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return (epoch + 1) / CONFIG['warmup_epochs']
        prog = (epoch - CONFIG['warmup_epochs']) / max(1, (proxy_epochs - CONFIG['warmup_epochs']))
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=CONFIG['learning_rate'] * 0.05)

    best = -1.0
    for epoch in range(proxy_epochs):
        # fixed pos-weight schedule
        t = min(epoch, CONFIG['weight_anneal_epochs'])
        pw = ((CONFIG['weight_anneal_epochs'] - t) / CONFIG['weight_anneal_epochs']) * \
             (CONFIG['weight_start'] - CONFIG['weight_end']) + CONFIG['weight_end']
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], device=device),
                                   ignore_index=-1)

        _ = train_one_epoch(model, tr_loader, crit, optimizer, scaler)

        if epoch < CONFIG['swa_start']:
            scheduler.step()
            ref = model
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            ref = swa_model

        score = eval_aupr(ref, vl_loader)
        best = max(best, score)
        trial.report(score, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best

# ---------------- Main ----------------
def set_deterministic(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Use 'spawn' to avoid inheriting open file descriptors into workers
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials")
    parser.add_argument("--proxy-epochs", type=int, default=18, help="epochs per trial (short)")
    parser.add_argument("--study-name", type=str, default="arch_search_gnn_int")
    parser.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///optuna.db")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| GPUs:", torch.cuda.device_count())
    print("CONFIG:", {k: v for k, v in CONFIG.items() if k not in ('sequence_file','data_file','edge_file')})

    set_deterministic(42)

    dataset = SafeSequenceParatopeDataset(
        sequence_file=CONFIG['sequence_file'],
        data_file=CONFIG['data_file'],
        edge_file=CONFIG['edge_file'],
        max_len=CONFIG['seq_len']
    )

    pruner = SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    study = optuna.create_study(direction="maximize",
                                study_name=args.study_name,
                                storage=args.storage,
                                load_if_exists=True,
                                pruner=pruner)
    study.optimize(lambda tr: objective(tr, dataset, args.proxy_epochs),
                   n_trials=args.trials, gc_after_trial=True)

    print("\nBest AUC-PR:", study.best_value)
    print("Best params:", study.best_params)

