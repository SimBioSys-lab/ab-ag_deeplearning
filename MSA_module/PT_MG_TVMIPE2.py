import math, os, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel

# ───────────────────────────────────────── CONFIG
config = {
    # data / model
    'sequence_file':  'MIPE_tv_esmsequences_1600.npz',
    'data_file':      'MIPE_tv_esminterfaces_1600.npz',
    'edge_file':      'MIPE_tv_esmedges_1600.npz',
    'vocab_size':     31,
    'seq_len':        1600,
    'embed_dim':      256,
    'num_heads':      16,
    'dropout':        0.1,
    'num_layers':     1,
    'num_gnn_layers': 8,
    'num_int_layers': 3,
    'drop_path_rate': 0.1,
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
    'n_splits':       10,
    'early_stop':     20,
    # pos-weight anneal
    'weight_start':   20.0,
    'weight_end':     1.0,
    'weight_anneal_epochs': 20,
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

# Define Focal Loss based on CrossEntropyLoss
class FocalLossCE(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean', ignore_index=-1):
        """
        Focal Loss for multi-class classification.

        Args:
            gamma (float): Focusing parameter.
            weight (Tensor, optional): A manual rescaling weight given to each class.
            reduction (str): 'mean' | 'sum' | 'none'
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(FocalLossCE, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Compute standard cross-entropy loss (per example)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight, ignore_index=self.ignore_index)
        # Compute the probability of the true class for each example
        pt = torch.exp(-ce_loss)
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ───────────────────────────────────────── Train / Val
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running = 0.0
    for i,(e,s,l) in enumerate(loader):
        e,s,l = e.to(device), s.to(device), l.to(device)
        if i % config['accum_steps']==0:
            optimizer.zero_grad(set_to_none=True)
        with autocast():
            out,_=model(sequences=s, padded_edges=e, return_attention=True)
            loss = criterion(out.view(-1,2), l.view(-1))/config['accum_steps']
        scaler.scale(loss).backward()
        if (i+1)%config['accum_steps']==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        running += loss.item()*config['accum_steps']
    return running/len(loader)
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    v=0.0
    for e,s,l in loader:
        e,s,l = e.to(device), s.to(device), l.to(device)
        out = model(sequences=s, padded_edges=e)
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
#    core_params = torch.load('isMIPEmodelnewesm_l1_g6_i3_do0.10_dpr0.10_lr0.0002_fold3_core.pth', map_location=device)
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
    ckpt=(f"iMIPEmodelnewesm_l{config['num_layers']}_g{config['num_gnn_layers']}"
          f"_i{config['num_int_layers']}_do{config['dropout']:.2f}"
          f"_dpr{config['drop_path_rate']:.2f}_lr{config['learning_rate']}"
          f"_fold{fold}.pth")
    torch.save(best_state, ckpt)
    print("Saved", ckpt)
