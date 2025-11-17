#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Antibody–antigen interface predictor
• warm-up → cosine until swa_start
• SWA weight averaging + small cosine inside SWA
"""

import os, math, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.model_selection import KFold
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew   import ClassificationModel

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ───────────────────────── CONFIG ──────────────────────────
cfg = {
    # data / model
    'sequence_file':  'cleaned_para_tv_sequences_1600.npz',
    'data_file':      'cleaned_para_tv_interfaces_1600.npz',
    'edge_file':      'cleaned_para_tv_edges_1600.npz',
    'vocab_size':     23,
    'seq_len':        1600,
    'embed_dim':      256,
    'num_heads':      16,
    'dropout':        0.10,
    'num_layers':     1,
    'num_gnn_layers': 12,
    'num_int_layers': 6,
    'drop_path_rate': 0.10,
    'num_classes':    2,

    # optimisation
    'batch_size':     4,
    'num_epochs':     50,
    'warmup_epochs':  10,
    'swa_start':      20,
    'learning_rate':  1e-4,
    'weight_decay':   1e-2,
    'max_grad_norm':  0.1,
    'accum_steps':    2,

    # early-stop & CV
    'n_splits':       5,
    'early_stop':     20,

    # class-weight anneal (20 → 2)
    'weight_start':   20.,
    'weight_end':     1.,
    'weight_anneal_epochs': 20,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(cfg)
print("GPUs available:", torch.cuda.device_count())

# ───────────────────────── dataloader ──────────────────────
def collate(batch):
    seqs, labels, edges = zip(*batch)          # each item: (1,L) , (L) , (M,2)
    seqs   = torch.stack(seqs)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    m      = max(e.shape[0] for e in edges)
    pads   = []
    for e in edges:
        pad = -torch.ones((2, m), dtype=torch.long)
        pad[:, :e.shape[0]] = e.T.clone().detach()
        pads.append(pad)
    edges  = torch.stack(pads)
    return edges, seqs, labels

# ────────────────────── train / validate ───────────────────
scaler = GradScaler()

def train_one_epoch(model, loader, crit, opt):
    model.train(); total = 0.
    for i,(e,s,l) in enumerate(loader):
        e,s,l = e.to(device), s.to(device), l.to(device)
        if i % cfg['accum_steps'] == 0:
            opt.zero_grad(set_to_none=True)
        with autocast():
            out,_ = model(sequences=s, padded_edges=e, return_attention=True)
            loss  = crit(out.view(-1,2), l.view(-1)) / cfg['accum_steps']
        scaler.scale(loss).backward()
        if (i+1) % cfg['accum_steps'] == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            scaler.step(opt); scaler.update()
        total += loss.item() * cfg['accum_steps']
    return total / len(loader)

@torch.no_grad()
def validate(model, loader, crit):
    model.eval(); tot = 0.
    for e,s,l in loader:
        e,s,l = e.to(device), s.to(device), l.to(device)
        out   = model(sequences=s, padded_edges=e)
        tot  += crit(out.view(-1,2), l.view(-1)).item()
    return tot / len(loader)

# ───────────────────────── dataset & CV ────────────────────
ds = SequenceParatopeDataset(
        cfg['data_file'], cfg['sequence_file'],
        cfg['edge_file'], cfg['seq_len'])
kf = KFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)

# small-cosine parameters
base_lr  = cfg['learning_rate']
swa_lr   = base_lr * 0.05
swa_low  = swa_lr * 0.70
swa_high = swa_lr * 1.30

def lr_factor_pre_swa(ep):
    """warm-up → cosine until swa_start-1 (down to swa_high)."""
    if ep < cfg['warmup_epochs']:
        return (ep+1)/cfg['warmup_epochs']
    prog   = (ep - cfg['warmup_epochs']) / (cfg['swa_start'] - cfg['warmup_epochs'])
    return 1.0 - (1.0 - swa_high/base_lr) * 0.5 * (1 + math.cos(math.pi*prog))

def lr_factor_swa(ep_after_swa):
    """small cosine inside SWA (±30 %)."""
    T = cfg['num_epochs'] - cfg['swa_start']
    cos = 0.5*(1+math.cos(math.pi*ep_after_swa/T))
    return (swa_low + (swa_high - swa_low)*cos) / base_lr

# ───────────────────────── main CV loop ────────────────────
for fold,(tr_idx,vl_idx) in enumerate(kf.split(np.arange(len(ds))), 1):
    print(f"\n===== Fold {fold}/{cfg['n_splits']} =====")
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=cfg['batch_size'],
                           shuffle=True, collate_fn=collate)
    vl_loader = DataLoader(Subset(ds, vl_idx), batch_size=cfg['batch_size'],
                           shuffle=False, collate_fn=collate)

    model = ClassificationModel(
                vocab_size=cfg['vocab_size'], seq_len=cfg['seq_len'],
                embed_dim=cfg['embed_dim'], num_heads=cfg['num_heads'],
                dropout=cfg['dropout'], num_layers=cfg['num_layers'],
                num_gnn_layers=cfg['num_gnn_layers'],
                num_int_layers=cfg['num_int_layers'],
                num_classes=cfg['num_classes'], drop_path_rate=cfg['drop_path_rate']).to(device)

    opt         = optim.AdamW(model.parameters(), lr=base_lr,
                              weight_decay=cfg['weight_decay'])
    pre_sched   = LambdaLR(opt, lr_lambda=lr_factor_pre_swa)
    swa_model   = AveragedModel(model)               # weight averaging only

    best_val = float('inf'); patience = 0
    for ep in range(cfg['num_epochs']):
        # dynamic pos-weight
        t  = min(ep, cfg['weight_anneal_epochs'])
        pw = ((cfg['weight_anneal_epochs'] - t)/cfg['weight_anneal_epochs']) * \
              (cfg['weight_start'] - cfg['weight_end']) + cfg['weight_end']
        crit = nn.CrossEntropyLoss(torch.tensor([1.0, pw], device=device),
                                   ignore_index=-1)

        tr = train_one_epoch(model, tr_loader, crit, opt)
        vl = validate(model, vl_loader, crit)

        # LR & SWA handling
        if ep < cfg['swa_start'] - 1:
            pre_sched.step()
        elif ep == cfg['swa_start'] - 1:
            pre_sched.step()                    # finish big cosine
            # set LR to swa_high for first SWA epoch (cosine starts high)
            for g in opt.param_groups: g['lr'] = swa_high
        else:
            # small cosine and weight averaging
            factor = lr_factor_swa(ep - cfg['swa_start'])
            for g in opt.param_groups: g['lr'] = base_lr * factor
            swa_model.update_parameters(model)

        cur_lr = opt.param_groups[0]['lr']
        print(f"Ep{ep:03d} | LR {cur_lr:.2e} | pw {pw:4.1f} | "
              f"train {tr:.4f} | val {vl:.4f}")

        eval_model = swa_model if ep >= cfg['swa_start'] else model
        eval_val   = validate(eval_model, vl_loader, crit)
        if eval_val < best_val:
            best_val, best_state = eval_val, eval_model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= cfg['early_stop']:
                print("Early stop triggered."); break

    # refresh BN + save
    if ep >= cfg['swa_start']:
        update_bn(tr_loader, swa_model)
        best_state = swa_model.state_dict()

    name = (f"iParamodelnew_l{cfg['num_layers']}_g{cfg['num_gnn_layers']}"
            f"_i{cfg['num_int_layers']}_do{cfg['dropout']:.2f}"
            f"_dpr{cfg['drop_path_rate']:.2f}_lr{cfg['learning_rate']}"
            f"_fold{fold}.pth")
    torch.save(best_state, name)
    print("Saved", name)

