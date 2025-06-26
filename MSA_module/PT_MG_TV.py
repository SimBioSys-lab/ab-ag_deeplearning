#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, math, logging, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew   import ClassificationModel

# ───────────────────────────────── CONFIG
cfg = {
    # data / model paths
    'sequence_file':  'MIPE_tv_sequences_1600.npz',
    'data_file':      'MIPE_tv_interfaces_1600.npz',
    'edge_file':      'MIPE_tv_edges_1600.npz',

    'vocab_size': 23,   'seq_len': 1600,
    'embed_dim':  256,  'num_heads': 16,
    'dropout':    0.25, 'num_layers': 0,
    'num_gnn_layers': 10, 'num_int_layers': 4,
    'drop_path_rate': 0.15,'num_classes': 2,

    # optimisation
    'batch_size': 4,     'num_epochs': 60,
    'warmup_epochs': 5,                        # ← swa_start removed
    'learning_rate': 1e-4,'weight_decay': 1e-2,
    'max_grad_norm': .1,  'accum_steps': 2,

    # early-stop & CV
    'n_splits': 5,       'early_stop': 15,

    # pos-weight anneal 10 → 1 over 40 epochs
    'weight_start': 1., 'weight_end': 1.,
    'weight_anneal_epochs': 10,
}

# ───────────────────────────────── SET-UP
ckpt_dir = (f"MIPE_l{cfg['num_layers']}_g{cfg['num_gnn_layers']}"
            f"_i{cfg['num_int_layers']}_do{cfg['dropout']:.2f}"
            f"_dpr{cfg['drop_path_rate']:.2f}_lr{cfg['learning_rate']}"
            "_checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(ckpt_dir, "log.txt"),
                                  mode="a", encoding="utf-8")],
)
log = logging.info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()
log("CONFIG " + str(cfg))
log(f"GPUs   {torch.cuda.device_count()}")

# ───────────────────────────────── Collate
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

# ───────────────────────────────── Train / Val
def train_one_epoch(model, loader, crit, opt):
    model.train(); run=0.0
    for i,(e,s,l) in enumerate(loader):
        e,s,l = e.to(device), s.to(device), l.to(device)
        if i % cfg['accum_steps']==0:
            opt.zero_grad(set_to_none=True)
        with autocast():
            out,_=model(sequences=s, padded_edges=e, return_attention=True)
            loss=crit(out.view(-1,2), l.view(-1))/cfg['accum_steps']
        scaler.scale(loss).backward()
        if (i+1)%cfg['accum_steps']==0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            scaler.step(opt); scaler.update()
        run+=loss.item()*cfg['accum_steps']
    return run/len(loader)

@torch.no_grad()
def validate(model, loader, crit):
    model.eval(); tot=0.0
    for e,s,l in loader:
        e,s,l = e.to(device), s.to(device), l.to(device)
        out   = model(sequences=s, padded_edges=e)
        tot  += crit(out.view(-1,2), l.view(-1)).item()
    return tot/len(loader)

# ───────────────────────────────── Dataset & CV
ds = SequenceParatopeDataset(cfg['data_file'],
                             cfg['sequence_file'],
                             cfg['edge_file'],
                             cfg['seq_len'])
kf = KFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)

# ───────────────────────────────── Main loop
for fold,(tr_idx,vl_idx) in enumerate(kf.split(np.arange(len(ds))),1):
    log(f"\n── Fold {fold}/{cfg['n_splits']} ──")

    tr_loader = DataLoader(Subset(ds,tr_idx), batch_size=cfg['batch_size'],
                           shuffle=True, collate_fn=custom_collate)
    vl_loader = DataLoader(Subset(ds,vl_idx), batch_size=cfg['batch_size'],
                           shuffle=False, collate_fn=custom_collate)

    model = ClassificationModel(
        vocab_size=cfg['vocab_size'], seq_len=cfg['seq_len'],
        embed_dim=cfg['embed_dim'], num_heads=cfg['num_heads'],
        dropout=cfg['dropout'], num_layers=cfg['num_layers'],
        num_gnn_layers=cfg['num_gnn_layers'], num_int_layers=cfg['num_int_layers'],
        num_classes=2, drop_path_rate=cfg['drop_path_rate']).to(device)
#    # Load parameters
#    core_params = torch.load('PPI_model_l0_g20_i8_do0.15_dpr0.15_lr0.0002_fold1_core.pth', map_location=device)
#    # Update model parameters
#    model_state = model.state_dict()
#    model_state.update(core_params)
#    model.load_state_dict(model_state)

    optim_ = optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                         weight_decay=cfg['weight_decay'])

    # warm-up → cosine (80 epochs)
    def lr_lambda(ep):
        if ep < cfg['warmup_epochs']:
            return (ep+1)/cfg['warmup_epochs']
        p = (ep-cfg['warmup_epochs'])/(cfg['num_epochs']-cfg['warmup_epochs'])
        return 0.5*(1+math.cos(math.pi*p))
    scheduler = LambdaLR(optim_, lr_lambda=lr_lambda)

    # ─── auto-resume
    prefix = f"fold{fold}_"
    ckpts  = sorted(f for f in os.listdir(ckpt_dir) if f.startswith(prefix))
    start_e, best_val, patience = 0, float("inf"), 0
    best_state = model.state_dict()                     # fallback
    if ckpts:
        last = os.path.join(ckpt_dir, ckpts[-1])
        log("Resuming from "+last)
        chk = torch.load(last, map_location=device)
        model.load_state_dict(chk['model'])
        optim_.load_state_dict(chk['optim'])
        scheduler.load_state_dict(chk['sched'])
        scaler.load_state_dict(chk['scaler'])
        start_e, best_val, patience = chk['epoch']+1, chk['best_val'], chk['patience']

    best_path = os.path.join(ckpt_dir, f"{prefix}best.pth")

    # ─── epoch loop
    for epoch in range(start_e, cfg['num_epochs']):
        # dynamic pos-weight
        t=min(epoch,cfg['weight_anneal_epochs'])
        pw=((cfg['weight_anneal_epochs']-t)/cfg['weight_anneal_epochs'])* \
           (cfg['weight_start']-cfg['weight_end'])+cfg['weight_end']
        crit = nn.CrossEntropyLoss(torch.tensor([1.0,pw],device=device),
                                   ignore_index=-1)

        tr= train_one_epoch(model, tr_loader, crit, optim_)
        vl= validate(model, vl_loader, crit)
        log(f"Fold {fold} | Ep {epoch:03d} | "
            f"LR={optim_.param_groups[0]['lr']:.2e} | "
            f"pw={pw:4.1f} | train={tr:.4f} | val={vl:.4f}")

        scheduler.step()

        # early stop on best-val
        if vl < best_val:
            best_val, best_state = vl, model.state_dict(); patience = 0
            torch.save(best_state, best_path)
            log(f"✓ new best (val={best_val:.4f}) → {best_path}")
        else:
            patience += 1
            if patience >= cfg['early_stop']:
                log("Early stopping"); break

        # rolling ckpt (keep last 3)
        roll = os.path.join(ckpt_dir, f"{prefix}e{epoch:03d}.pth")
        torch.save({'epoch':epoch,'model':model.state_dict(),
                    'optim':optim_.state_dict(),'sched':scheduler.state_dict(),
                    'scaler':scaler.state_dict(),
                    'best_val':best_val,'patience':patience}, roll)
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.startswith(prefix))
        for old in ckpts[:-3]:
            os.remove(os.path.join(ckpt_dir, old))

    # ─── save final best
    torch.save(best_state, best_path)          # ensure best snapshot exists
    final = (f"iPPIMIPEmodel_l{cfg['num_layers']}_g{cfg['num_gnn_layers']}"
             f"_i{cfg['num_int_layers']}_do{cfg['dropout']:.2f}"
             f"_dpr{cfg['drop_path_rate']:.2f}_lr{cfg['learning_rate']}"
             f"_fold{fold}.pth")
    torch.save(best_state, final)
    log("Saved final best → "+final)

