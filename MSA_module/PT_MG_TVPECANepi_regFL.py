import math, os, torch, numpy as np, random
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler  # ← deprecation-safe
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score  # ← for AUPR
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel

# ───────────────────────────────────────── SEEDS
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(42)

# ───────────────────────────────────────── CONFIG (tuned)
config = {
    # data / model
    'sequence_file':  'pecanepi_tv_esmsequences_1600.npz',
    'data_file':      'pecanepi_tv_esminterfaces_1600.npz',
    'edge_file':      'pecanepi_tv_esmedges_1600.npz',
    'vocab_size':     31,
    'seq_len':        1600,
    'embed_dim':      256,
    'num_heads':      4,
    'dropout':        0.1,
    'num_layers':     1,
    'num_gnn_layers': 4,
    'num_int_layers': 2,
    'drop_path_rate': 0.10,
    'num_classes':    2,

    # optimisation
    'batch_size':     4,
    'num_epochs':     60,
    'warmup_epochs':  10,
    'swa_start':      20,
    'learning_rate':  2e-4,
    'weight_decay':   3e-4,
    'max_grad_norm':  0.5,
    'accum_steps':    2,

    # early-stop & CV
    'n_splits':       5,
    'early_stop':     15,

    # pos-weight anneal
    'weight_start':   20.0,
    'weight_end':     1.0,
    'weight_anneal_epochs': 20,

    # graph-specific regularization (base values)
    'dropedge_min':   0.01,
    'dropedge_max':   0.05,
    'token_drop_p':   0.05,   # on indices, not embeddings; PAD=1
    'apply_feature_mask_on_train_only': True,

    # loss
    'label_smoothing': 0.0,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# ───────────────────────────────────────── Collate
def custom_collate_fn(batch):
    seqs, labels, edges = zip(*batch)
    seqs   = torch.stack(seqs)                               # [B, M, L] OR [B, L]; your model handles it
    labels = torch.tensor(np.array(labels), dtype=torch.long)  # [B, L]
    max_e  = max(e.shape[0] for e in edges)
    pads   = []
    for e in edges:
        pad = -torch.ones((2, max_e), dtype=torch.long)      # -1 marks padded “no-edge”
        pad[:, :e.shape[0]] = e.T.clone().detach()
        pads.append(pad)
    edges = torch.stack(pads)                                # [B, 2, Emax]
    return edges, seqs, labels

# ───────────────────────────────────────── Graph-specific regularizers
def random_p(min_p, max_p):
    return min_p + (max_p - min_p) * random.random()

def dropedge_padded(padded_edges, p=0.05):
    if p <= 0: return padded_edges
    B, _, Emax = padded_edges.shape
    pe = padded_edges.clone()
    for b in range(B):
        col = pe[b]
        valid_mask = (col[0] >= 0) & (col[1] >= 0)
        idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0: continue
        k = int(p * idx.numel())
        if k <= 0: continue
        drop_idx = idx[torch.randperm(idx.numel(), device=pe.device)[:k]]
        pe[b, :, drop_idx] = -1
    return pe

def token_dropout(seqs, p=0.05, pad_id=1):
    """Randomly replace tokens with PAD in [B, M, L] indices. Protect query row M=0."""
    if p <= 0:
        return seqs
    s = seqs.clone()
    mask = (torch.rand_like(s.float()) < p) & (s != pad_id) & (s != -1)
    # protect the query sequence (row 0 along M)
    if s.dim() == 3:  # [B, M, L]
        mask[:, 0, :] = False
    s[mask] = pad_id
    return s

# ───────────────────────────────────────── Focal Loss
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    Args:
        alpha (float): scaling factor for positive examples (default=1.0)
        gamma (float): focusing parameter (default=2.0)
        weight (torch.Tensor): optional class weights [num_classes]
        ignore_index (int): index to ignore (like -1)
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N]
        ce_loss = self.ce(logits, targets)  # per-sample CE
        pt = torch.exp(-ce_loss)            # pt = prob(correct)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # mask ignored targets
        valid_mask = targets != self.ignore_index
        return focal_loss[valid_mask].mean() if valid_mask.any() else focal_loss.mean()

# ───────────────────────────────────────── Train / Val
def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    token_p, dropedge_min, dropedge_max):
    model.train()
    running = 0.0
    for i,(e,s,l) in enumerate(loader):
        e = e.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        l = l.to(device, non_blocking=True)

        # Graph-specific regularization (train-time only; ramped)
        e = dropedge_padded(e, p=random_p(dropedge_min, dropedge_max))
        if config['apply_feature_mask_on_train_only']:
            s = token_dropout(s, p=token_p, pad_id=1)

        if i % config['accum_steps']==0:
            optimizer.zero_grad(set_to_none=True)

        with autocast():
            out,_ = model(sequences=s, padded_edges=e, return_attention=True)
            loss = criterion(out.view(-1,2), l.view(-1)) / config['accum_steps']

        scaler.scale(loss).backward()
        if (i+1)%config['accum_steps']==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()

        running += loss.item()*config['accum_steps']
    return running/len(loader)

@torch.no_grad()
def validate_with_aupr(model, loader, criterion):
    """Return (loss, AUPR) computed over all tokens with labels != -1."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    for e,s,l in loader:
        e = e.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        l = l.to(device, non_blocking=True)

        logits = model(sequences=s, padded_edges=e)  # [B, L, 2] or compatible
        loss = criterion(logits.view(-1,2), l.view(-1))
        total_loss += loss.item()

        # collect probs for positive class
        probs = torch.softmax(logits, dim=-1)[..., 1]  # [B, L]
        all_probs.append(probs.reshape(-1).detach().cpu())
        all_labels.append(l.reshape(-1).detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # mask ignore_index (-1)
    mask = all_labels != -1
    if mask.sum() == 0:
        aupr = 0.0
    else:
        aupr = average_precision_score(all_labels[mask], all_probs[mask])

    return total_loss/len(loader), aupr

# ───────────────────────────────────────── Dataset & CV
dataset = SequenceParatopeDataset(
    data_file=config['data_file'],
    sequence_file=config['sequence_file'],
    edge_file=config['edge_file'],
    max_len=config['seq_len'])

kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)

# ───────────────────────────────────────── Main loop
scaler = GradScaler('cuda')
for fold,(tr_idx,vl_idx) in enumerate(kf.split(np.arange(len(dataset))),1):
    print(f"\n── Fold {fold}/{config['n_splits']}")
    tr_loader=DataLoader(Subset(dataset,tr_idx),
        batch_size=config['batch_size'], shuffle=True,
        collate_fn=custom_collate_fn)
    vl_loader=DataLoader(
        Subset(dataset,vl_idx),
        batch_size=config['batch_size'], shuffle=False,
        collate_fn=custom_collate_fn)

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

    # parameter groups (core vs others) – keep your logic
    base_lr = config['learning_rate']
    core_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "mc_model" in name:
            core_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": core_params,  "lr": base_lr * 1.0},
            {"params": other_params, "lr": base_lr * 1.0},
        ],
        lr=base_lr,
        weight_decay=config['weight_decay']
    )

    # warm-up → cosine
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch+1)/config['warmup_epochs']
        progress = (epoch-config['warmup_epochs'])/max(1,(config['num_epochs']-config['warmup_epochs']))
        return 0.5*(1+math.cos(math.pi*progress))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # SWA
    swa_model      = AveragedModel(model)
    swa_scheduler  = SWALR(optimizer, swa_lr=config['learning_rate']*0.05)

    # Early stop on VAL AUPR (better than loss for imbalance)
    best_aupr = -1.0
    patience = 0
    best_state = None

    for epoch in range(config['num_epochs']):
        # pos-weight anneal (20→1 over first 20 epochs)
        t = min(epoch, config['weight_anneal_epochs'])
        pw = ((config['weight_anneal_epochs'] - t) / config['weight_anneal_epochs']) * \
             (config['weight_start'] - config['weight_end']) + config['weight_end']
        class_weights = torch.tensor([1.0, pw], device=device)
        
        criterion = FocalLoss(
            alpha=1.0, gamma=2.0,
            weight=class_weights,
            ignore_index=-1
        )
        # ── Regularization ramp: start light, increase after epoch 15 over ~10 epochs
        if epoch <= 15:
            phase = 0.0
        else:
            phase = min(1.0, (epoch - 15) / 10.0)  # ramp 0→1 from epoch 16 to 25
        curr_token_p  = config['token_drop_p'] * phase
        curr_drop_min = config['dropedge_min'] * phase
        curr_drop_max = config['dropedge_max'] * phase

        tr_loss = train_one_epoch(model, tr_loader, criterion, optimizer, scaler,
                                  token_p=curr_token_p,
                                  dropedge_min=curr_drop_min,
                                  dropedge_max=curr_drop_max)

        # scheduler / SWA
        if epoch < config['swa_start']:
            scheduler.step()
            eval_model = model
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            eval_model = swa_model

        vl_loss, vl_aupr = validate_with_aupr(eval_model, vl_loader, criterion)
        print(f"Ep{epoch:03d}  LR={optimizer.param_groups[0]['lr']:.2e}  "
              f"pw={pw:.1f}  token_p={curr_token_p:.3f}  dropE=({curr_drop_min:.3f},{curr_drop_max:.3f})  "
              f"train={tr_loss:.4f}  val_loss={vl_loss:.4f}  val_AUPR={vl_aupr:.4f}")

        # Early stop on AUPR
        if vl_aupr > best_aupr + 1e-4:
            best_aupr = vl_aupr
            patience = 0
            # keep best params from the model you're evaluating (SWA or base)
            best_state = eval_model.state_dict()
        else:
            patience += 1
            if patience >= config['early_stop']:
                print("Early stop triggered (val AUPR)")
                break

    # final BN update & save
    if epoch >= config['swa_start']:
        update_bn(tr_loader, swa_model)
        best_state = swa_model.state_dict() if best_state is None else best_state

    ckpt=(f"iPECANepiregFL_l{config['num_layers']}_g{config['num_gnn_layers']}"
          f"_i{config['num_int_layers']}_do{config['dropout']:.2f}"
          f"_dpr{config['drop_path_rate']:.2f}_lr{config['learning_rate']}"
          f"_heads{config['num_heads']}_fold{fold}.pth")
    torch.save(best_state, ckpt)
    print(f"Saved {ckpt}  (best val AUPR={best_aupr:.4f})")

