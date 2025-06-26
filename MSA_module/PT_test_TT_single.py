import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    precision_recall_curve, auc, f1_score
)
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel
from torch.amp import autocast
import os

# Configuration
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Read model filenames
with open('tt45model_files', 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# Test configuration
test_config = {
    'batch_size': 1,
    'sequence_file': 'cleaned_para_test_sequences_1600.npz',
    'data_file': 'cleaned_para_test_interfaces_1600.npz',
    'edge_file': 'cleaned_para_test_edges_1600.npz',
    'max_len': 1600,
    'vocab_size': 23,
    'num_classes': 2
}

def custom_collate_fn(batch):
    sequences, pt, edge = zip(*batch)
    sequence_tensor = torch.stack(sequences)                  # (B, 1, L)
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long) # (B, L)
    max_edges = max(e.shape[0] for e in edge)
    padded = []
    for e in edge:
        pad = -torch.ones((2, max_edges), dtype=torch.long)
        pad[:, :e.shape[0]] = torch.from_numpy(e.T)
        padded.append(pad)
    padded_edges = torch.stack(padded)                        # (B, 2, E)
    return padded_edges, sequence_tensor, pt_tensor

# Load test dataset
test_ds = SequenceParatopeDataset(
    data_file=test_config['data_file'],
    sequence_file=test_config['sequence_file'],
    edge_file=test_config['edge_file'],
    max_len=test_config['max_len']
)
test_loader = DataLoader(
    test_ds,
    batch_size=test_config['batch_size'],
    collate_fn=custom_collate_fn,
    shuffle=False
)

# Load interface keys
interfaces_npz = np.load(test_config['data_file'], allow_pickle=True)
interface_keys = list(interfaces_npz.keys())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_file in models:
    torch.cuda.empty_cache()
    print(f"\n=== Testing model: {model_file} ===")

    # extract hyperparams from filename
    parts = model_file.split('_')
    num_layers     = int(parts[1][1:])
    num_gnn_layers = int(parts[2][1:])
    num_int_layers = int(parts[3][1:])
    embed_dim      = 256
    num_heads      = 16
    drop_path_rate = 0.1
    dropout        = float(parts[4][2:-4])

    # build and load model
    model = ClassificationModel(
        vocab_size=test_config['vocab_size'],
        seq_len=test_config['max_len'],
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        num_layers=num_layers,
        num_gnn_layers=num_gnn_layers,
        num_int_layers=num_int_layers,
        num_classes=test_config['num_classes']
    ).to(device)
    ckpt = torch.load(model_file, map_location=device)
    if next(iter(ckpt)).startswith('module.'):
        ckpt = {k[len('module.'):]: v for k,v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()

    # per-sample evaluation
    with torch.no_grad():
        for sample_idx, (padd, seqs, pts) in enumerate(test_loader):
            padd = padd.to(device)
            seqs = seqs.to(device)
            pts  = pts.to(device)            # (1, L)

            with autocast(device_type='cuda'):
                outputs, _ = model(
                    sequences=seqs,
                    padded_edges=padd,
                    return_attention=True
                )
            outputs = outputs.squeeze(1)      # (1, L, 2)  or (1,2,L)
            # ensure shape is (1, L, 2)
            if outputs.dim()==3 and outputs.shape[1]==2:
                outputs = outputs.permute(0,2,1)  # (1,L,2)
            probs = torch.softmax(outputs, dim=-1)  # (1,L,2)

            # flatten valid positions
            valid = pts[0] >= 0
            true_labels = pts[0][valid].cpu().numpy()
            pos_probs   = probs[0, valid, 1].cpu().numpy()

            # default preds at 0.5
            preds_05 = (pos_probs > 0.5).astype(int)

            # compute per-sample metrics
            acc     = (preds_05 == true_labels).mean()
            prec    = precision_score(true_labels, preds_05, zero_division=0)
            rec     = recall_score   (true_labels, preds_05, zero_division=0)
            f1      = f1_score       (true_labels, preds_05, zero_division=0)
            try:
                auc_roc = roc_auc_score(true_labels, pos_probs)
            except ValueError:
                auc_roc = float('nan')
            p_curve, r_curve, _ = precision_recall_curve(true_labels, pos_probs)
            auc_pr = auc(r_curve, p_curve)

            key = interface_keys[sample_idx] if sample_idx < len(interface_keys) else f"idx_{sample_idx}"
            print(f"Sample {sample_idx} ({key}): "
                  f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  "
                  f"F1={f1:.3f}  AUC-ROC={auc_roc:.3f}  AUC-PR={auc_pr:.3f}")

    # (optionally) aggregated metrics can follow here…

