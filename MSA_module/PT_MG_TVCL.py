import math, os, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from Dataloader_itf import SequenceParatopeDataset
from Models_fullCL import ClassificationModel

# ───────────────────────────────────────── CONFIG
config = {
    # data / model
    'sequence_file':  'cleaned_para_tv_sequences_1600.npz',
    'data_file':      'cleaned_para_tv_interfaces_1600.npz',
    'edge_file':      'cleaned_para_tv_edges_1600.npz',
    'vocab_size':     23,
    'seq_len':        1600,
    'embed_dim':      256,
    'num_heads':      16,
    'dropout':        0.1,
    'num_layers':     0,
    'num_gnn_layers': 20,
    'num_int_layers': 8,
    'drop_path_rate': 0.1,
    'num_classes':    2,
    # optimisation
    'batch_size':     4,
    'num_epochs':     50,
    'warmup_epochs':  10,
    'swa_start':      20,
    'learning_rate':  2e-4,
    'weight_decay':   1e-2,
    'max_grad_norm':  0.1,
    'accum_steps':    2,
    # early-stop & CV
    'n_splits':       5,
    'early_stop':     20,
    # pos-weight anneal
    'weight_start':   20.0,
    'weight_end':     1.0,
    'weight_anneal_epochs': 10,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# ───────────────────────────────────────── Collate
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

# ───────────────────────────────────────── Train / Val
alpha      = 0.0001          # mix-factor for contrastive loss
tau        = 0.5          # temperature
eps        = 1e-6
pos_weight = 20.0         # imbalance 1:20 → up-weight positives

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
def validate(model, loader, criterion):
    model.eval()
    v=0.0
    for e,s,l in loader:
        e,s,l = e.to(device), s.to(device), l.to(device)
        out, _ = model(sequences=s, padded_edges=e)
        v  += criterion(out.view(-1,2), l.view(-1)).item()
    return v/len(loader)

# ───────────────────────────────────────── Dataset & CV
dataset = SequenceParatopeDataset(
    data_file=config['data_file'],
    sequence_file=config['sequence_file'],
    edge_file=config['edge_file'],
    max_len=config['seq_len'])
kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)

# ───────────────────────────────────────── Main loop
scaler = GradScaler()
for fold,(tr_idx,vl_idx) in enumerate(kf.split(np.arange(len(dataset))),1):
    print(f"\n── Fold {fold}/{config['n_splits']}")
    tr_loader=DataLoader(Subset(dataset,tr_idx),batch_size=config['batch_size'],
                         shuffle=True,collate_fn=custom_collate_fn)
    vl_loader=DataLoader(Subset(dataset,vl_idx),batch_size=config['batch_size'],
                         shuffle=False,collate_fn=custom_collate_fn)

    model = ClassificationModel(
        vocab_size=config['vocab_size'], seq_len=config['seq_len'],
        embed_dim=config['embed_dim'], num_heads=config['num_heads'],
        dropout=config['dropout'], num_layers=config['num_layers'],
        num_gnn_layers=config['num_gnn_layers'], num_int_layers=config['num_int_layers'],
        num_classes=2, drop_path_rate=config['drop_path_rate'])
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)
#    # Load parameters
#    core_params = torch.load('isParamodel_l0_g20_i8_dp0.1_core_module.pth', map_location=device)
#    # Update model parameters
#    model_state = model.state_dict()
#    model_state.update(core_params)
#    model.load_state_dict(model_state)


    # two param groups so we could freeze something later if desired
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])

    # warm-up → cosine schedule (group-wise)
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch+1)/config['warmup_epochs']
        progress = (epoch-config['warmup_epochs'])/(config['num_epochs']-config['warmup_epochs'])
        return 0.5*(1+math.cos(math.pi*progress))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # SWA objects
    swa_model      = AveragedModel(model)
    swa_scheduler  = SWALR(optimizer, swa_lr=config['learning_rate']*0.05)

    best_val=float('inf'); patience=0
    for epoch in range(config['num_epochs']):
        # dynamic pos-weight (20→1 over first 20 epochs)
        t=min(epoch,config['weight_anneal_epochs'])
        pw = ((config['weight_anneal_epochs']-t)/config['weight_anneal_epochs'])* \
             (config['weight_start']-config['weight_end']) + config['weight_end']
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,pw], device=device),
                                        ignore_index=-1)

        tr_loss=train_one_epoch(model,tr_loader,criterion,optimizer,scaler)
        vl_loss=validate(model,vl_loader,criterion)
        print(f"Ep{epoch:03d}  LR={optimizer.param_groups[0]['lr']:.2e} "
              f"pw={pw:.1f}  train={tr_loss:.4f}  val={vl_loss:.4f}")
        # scheduler / SWA
        if epoch < config['swa_start']:
            scheduler.step()
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # early stop on *SWA* val if we're past swa_start, else on model val
        eval_val = validate(swa_model if epoch>=config['swa_start'] else model,
                            vl_loader, criterion)
        if eval_val < best_val:
            best_val,best_state = eval_val, \
                (swa_model if epoch>=config['swa_start'] else model).state_dict()
            patience=0
        else:
            patience+=1
            if patience>=config['early_stop']:
                print("Early stop triggered")
                break

    # final BN update & save
    if epoch>=config['swa_start']:
        update_bn(tr_loader, swa_model)
        best_state = swa_model.state_dict()
    ckpt=(f"iParamodelCL_l{config['num_layers']}_g{config['num_gnn_layers']}"
          f"_i{config['num_int_layers']}_do{config['dropout']:.2f}"
          f"_dpr{config['drop_path_rate']:.2f}_lr{config['learning_rate']}"
          f"_fold{fold}.pth")
    torch.save(best_state, ckpt)
    print("Saved", ckpt)
