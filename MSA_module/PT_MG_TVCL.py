#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training with   warm-up ➜ cosine-decay  (for **all** epochs, even in SWA phase)
and Stochastic-Weight-Averaging (only weight averaging, no SWALR).

• 5-fold CV, early-stopping, AMP, gradient-accumulation
• positive-class weight decays 20 → 1 over 50 epochs
"""

import math, os, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from Dataloader_itf import SequenceParatopeDataset
from Models_fullCL   import ClassificationModel
import torch.nn.functional as F
# ───────────────────────────────── CONFIG
cfg = {
    # data / model
    'sequence_file':  'cleaned_para_tv_sequences_1600.npz',
    'data_file':      'cleaned_para_tv_interfaces_1600.npz',
    'edge_file':      'cleaned_para_tv_edges_1600.npz',

    'vocab_size':     23,
    'seq_len':        1600,
    'embed_dim':      256,
    'num_heads':      16,
    'dropout':        0.2,
    'num_layers':     0,
    'num_gnn_layers': 20,
    'num_int_layers': 8,
    'drop_path_rate': 0.2,
    'num_classes':    2,

    # optimisation
    'batch_size':     4,
    'num_epochs':     60,
    'warmup_epochs':  5,
    'swa_start':      30,
    'learning_rate':  1e-4,
    'weight_decay':   1e-2,
    'max_grad_norm':  0.1,
    'accum_steps':    2,

    # early stop & CV
    'n_splits':       5,
    'early_stop':     20,

    # pos-weight schedule
    'weight_start':   1.0,
    'weight_end':     1.0,
    'weight_anneal_epochs': 20,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(cfg)
print("GPUs :", torch.cuda.device_count())

# ───────────────────────────────── collate
def custom_collate(batch):
    seqs, labels, edges = zip(*batch)
    seqs   = torch.stack(seqs)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    max_e  = max(e.shape[0] for e in edges)
    pads   = []
    for e in edges:
        pad = -torch.ones((2, max_e), dtype=torch.long)
        pad[:, :e.shape[0]] = e.T.clone().detach()
        pads.append(pad)
    return torch.stack(pads), seqs, labels

# ───────────────────────────────── train / val
alpha      = 0.01          # mix-factor for contrastive loss
tau        = 0.5          # temperature
eps        = 1e-6
pos_weight = 1.0         # imbalance 1:20 → up-weight positives

def train_one_epoch(model, loader, ce_crit, optim_):
    model.train()
    run_loss = 0.0

    for step, (edges, seqs, labels) in enumerate(loader):
        edges, seqs, labels = edges.to(device), seqs.to(device), labels.to(device)

        if step % cfg['accum_steps'] == 0:
            optim_.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            logits, emb, _ = model(sequences=seqs,
                                   padded_edges=edges,
                                   return_attention=True)

            # ───── cross-entropy (class-weighted) ─────
            ce_loss = ce_crit(logits.view(-1, 2), labels.view(-1))

            # ───── contrastive part ─────
            z = emb.view(-1, emb.size(-1))          # (B*L, D)
            y = labels.view(-1)                     # (B*L,)

            keep = y != -1                          # drop ignore label
            z, y = z[keep], y[keep]
            m = z.size(0)

            if m > 1:
                z  = F.normalize(z, dim=1)
                cos = torch.matmul(z, z.T)          # (m,m)
                cos.fill_diagonal_(0.0)

                same = y[:, None] == y[None, :]
                diff = ~same

                s   = torch.exp(cos / tau)          # similarity = e^{cos/τ}

                denom = (s * diff.float()).sum(1, keepdim=True) + eps  # (m,1)

                # log-ratio for every positive pair (i,j)
                log_ratio = torch.log(s / denom)    # (m,m)

                pair_mask = same.float()
                pair_mask.fill_diagonal_(0.)

                # optional class weight on i-index (row)
                row_w = torch.where(y == 1,
                                    z.new_tensor(pos_weight),
                                    z.new_tensor(1.0)).view(-1, 1)
                con_loss = -(log_ratio * pair_mask * row_w).sum() / m
            else:
                con_loss = torch.tensor(0., device=device)

            loss = (ce_loss + alpha * con_loss) / cfg['accum_steps']

        scaler.scale(loss).backward()

        if (step + 1) % cfg['accum_steps'] == 0:
            scaler.unscale_(optim_)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            scaler.step(optim_)
            scaler.update()

        run_loss += loss.item() * cfg['accum_steps']

    return run_loss / len(loader)

@torch.no_grad()
def validate(model, loader, crit):
    model.eval(); tot=0.0
    for e,s,l in loader:
        e,s,l = e.to(device), s.to(device), l.to(device)
        out, embeddings   = model(sequences=s, padded_edges=e)
        tot  += crit(out.view(-1,2), l.view(-1)).item()
    return tot/len(loader)

# ───────────────────────────────── dataset & folds
ds = SequenceParatopeDataset(cfg['data_file'], cfg['sequence_file'],
                             cfg['edge_file'], cfg['seq_len'])
kf = KFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
scaler = GradScaler()

# ───────────────────────────────── main loop
for fold,(tr_idx,vl_idx) in enumerate(kf.split(np.arange(len(ds))),1):
    print(f"\n── Fold {fold}/{cfg['n_splits']} ──")
    tr_loader = DataLoader(Subset(ds,tr_idx), batch_size=cfg['batch_size'],
                           shuffle=True, collate_fn=custom_collate)
    vl_loader = DataLoader(Subset(ds,vl_idx), batch_size=cfg['batch_size'],
                           shuffle=False, collate_fn=custom_collate)

    model = ClassificationModel(
        vocab_size=cfg['vocab_size'], seq_len=cfg['seq_len'],
        embed_dim=cfg['embed_dim'], num_heads=cfg['num_heads'],
        dropout=cfg['dropout'], num_layers=cfg['num_layers'],
        num_gnn_layers=cfg['num_gnn_layers'], num_int_layers=cfg['num_int_layers'],
        num_classes=2, drop_path_rate=cfg['drop_path_rate']
    ).to(device)
#    # Load parameters
#    core_params = torch.load('PPI_model_l0_g20_i8_do0.15_dpr0.15_lr0.0002_fold1_core.pth', map_location=device)
#    # Update model parameters
#    model_state = model.state_dict()
#    model_state.update(core_params)
#    model.load_state_dict(model_state)

    optim_ = optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                         weight_decay=cfg['weight_decay'])
    # warm-up ➜ cosine entire 80 epochs
    def lr_lambda(ep):
        if ep < cfg['warmup_epochs']:
            return (ep+1)/cfg['warmup_epochs']
        p = (ep-cfg['warmup_epochs'])/(cfg['num_epochs']-cfg['warmup_epochs'])
        return 0.5*(1+math.cos(math.pi*p))
    scheduler = LambdaLR(optim_, lr_lambda=lr_lambda)

    # SWA (only averaging, no LR control)
    swa_model = AveragedModel(model)

    best_val=float('inf'); patience=0
    for epoch in range(cfg['num_epochs']):
        # dynamic class weight
        t=min(epoch,cfg['weight_anneal_epochs'])
        pw=((cfg['weight_anneal_epochs']-t)/cfg['weight_anneal_epochs'])* \
            (cfg['weight_start']-cfg['weight_end'])+cfg['weight_end']
        crit = nn.CrossEntropyLoss(torch.tensor([1.0,pw],device=device),
                                   ignore_index=-1)
        tr = train_one_epoch(model,tr_loader,crit,optim_)
        vl = validate(model,vl_loader,crit)
        print(f"Ep{epoch:03d}  LR={optim_.param_groups[0]['lr']:.2e}  "
              f"pw={pw:4.1f}  train={tr:.4f}  val={vl:.4f}")

        scheduler.step()                     # ← keep cosine forever

        # update SWA model after swa_start
        if epoch >= cfg['swa_start']:
            swa_model.update_parameters(model)

        # early stopping logic
        eval_m = swa_model if epoch>=cfg['swa_start'] else model
        eval_v = validate(eval_m, vl_loader, crit)
        if eval_v < best_val:
            best_val, best_state = eval_v, eval_m.state_dict(); patience=0
        else:
            patience += 1
            if patience >= cfg['early_stop']:
                print("Early stop"); break

#    # batch-norm refresh & save
#    if epoch >= cfg['swa_start']:
#        update_bn(tr_loader, swa_model)
#        best_state = swa_model.state_dict()

    name = (f"iParaCLmodel_l{cfg['num_layers']}_g{cfg['num_gnn_layers']}"
            f"_i{cfg['num_int_layers']}_do{cfg['dropout']:.2f}"
            f"_dpr{cfg['drop_path_rate']:.2f}_lr{cfg['learning_rate']}"
            f"_fold{fold}.pth")
    torch.save(best_state, name)
    print("Saved", name)

