import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score
)
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel
from torch.amp import autocast
import csv
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

# Custom collate function
def custom_collate_fn(batch):
    sequences, pt, edge = zip(*batch)
    sequence_tensor = torch.stack(sequences)
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long)
    max_edges = max(edge_index.shape[0] for edge_index in edge)
    padded_edges = []
    for edge_index in edge:
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, :edge_index.shape[0]] = edge_index.T.clone().detach()
        padded_edges.append(edge_pad)
    return torch.stack(padded_edges), sequence_tensor, pt_tensor

# Load the dataset
test_dataset = SequenceParatopeDataset(
    data_file=test_config['data_file'],
    sequence_file=test_config['sequence_file'],
    edge_file=test_config['edge_file'],
    max_len=test_config['max_len']
)
test_loader = DataLoader(
    test_dataset,
    batch_size=test_config['batch_size'],
    collate_fn=custom_collate_fn,
    shuffle=False
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the interfaces NPZ to get sample keys
interfaces_npz = np.load(test_config['data_file'], allow_pickle=True)
interface_keys = list(interfaces_npz.keys())

# Test each model
for model_file in models:
    torch.cuda.empty_cache()
    print(f"\nTesting model: {model_file}")

    # Parse configuration from filename
    parts = model_file.split('_')
    num_layers     = int(parts[1][1:])
    num_gnn_layers = int(parts[2][1:])
    num_int_layers = int(parts[3][1:])
    embed_dim      = 256
    num_heads      = 16
    drop_path_rate = 0.1
    dropout        = float(parts[4][2:-4])

    # Initialize model
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

    # Load weights
    checkpoint = torch.load(model_file, map_location=device)
    # strip `module.` if necessary
    if next(iter(checkpoint)).startswith('module.'):
        checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()

    # Prepare metrics storage
    metrics = {
        "Lchain":   {"true": [], "probs": []},
        "Hchain":   {"true": [], "probs": []},
        "Antibody": {"true": [], "probs": []},
        "Antigen":  {"true": [], "probs": []},
    }

    # Run inference
    with torch.no_grad():
        for sample_idx, (padded_edges, sequence_tensor, pt_tensor) in enumerate(test_loader):
            padded_edges   = padded_edges.to(device)
            sequence_tensor= sequence_tensor.to(device)
            pt_tensor      = pt_tensor.to(device)

            with autocast(device_type='cuda'):
                outputs, _ = model(
                    sequences=sequence_tensor,
                    padded_edges=padded_edges,
                    return_attention=True
                )
            outputs = outputs.squeeze(1)              # [B, 2, L, L]
            probs   = torch.softmax(outputs, dim=1)   # [B, 2, L, L]

            # find segment splits
            seq_arr = sequence_tensor[0, 0].cpu().numpy()
            EOCs = np.where(seq_arr == 22)[0]
            chain_types  = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(EOCs)-2)]
            chain_splits = [0] + EOCs.tolist()

            # collect per-chain metrics
            for i, chain_type in enumerate(chain_types):
                start, end = chain_splits[i], chain_splits[i+1]
                mask = pt_tensor[0, start:end] >= 0
                if not mask.any():
                    continue

                true_labels = pt_tensor[0, start:end][mask].cpu().numpy()
                pos_probs   = probs[0, 1, start:end][mask].cpu().numpy()

                metrics[chain_type]["true"].extend(true_labels.tolist())
                metrics[chain_type]["probs"].extend(pos_probs.tolist())

                # antibody vs antigen grouping
                group = "Antibody" if chain_type in ("Lchain","Hchain") else "Antigen"
                metrics[group]["true"].extend(true_labels.tolist())
                metrics[group]["probs"].extend(pos_probs.tolist())

    # Compute and print metrics, including optimal threshold sweep
    for chain_type, data in metrics.items():
        true = np.array(data["true"])
        probs = np.array(data["probs"])
        if true.size == 0:
            print(f"No data for {chain_type}. Skipping.")
            continue

        # Default (0.5) threshold
        default_preds = (probs > 0.5).astype(int)

        # Standard metrics
        acc   = (default_preds == true).mean()
        prec  = precision_score(true, default_preds, zero_division=0)
        rec   = recall_score   (true, default_preds, zero_division=0)
        auc_roc = roc_auc_score(true, probs)
        p_curve, r_curve, _ = precision_recall_curve(true, probs)
        auc_pr = auc(r_curve, p_curve)

        print(f"\n=== {chain_type} (default threshold=0.50) ===")
        print(f" Accuracy: {acc:.4f}")
        print(f" Precision: {prec:.4f}")
        print(f" Recall:    {rec:.4f}")
        print(f" AUC-ROC:   {auc_roc:.4f}")
        print(f" AUC-PR:    {auc_pr:.4f}")

        # Sweep threshold for best F1
        p_curve, r_curve, thresholds = precision_recall_curve(true, probs)
        f1_scores = 2 * (p_curve * r_curve) / (p_curve + r_curve + 1e-8)
        # drop last point where threshold is undefined
        best_idx     = np.argmax(f1_scores[:-1])
        best_thresh  = thresholds[best_idx]
        best_f1      = f1_scores[best_idx]
        best_prec    = p_curve[best_idx]
        best_rec     = r_curve[best_idx]

        print(f"→ Best threshold for F1: {best_thresh:.3f}")
        print(f"   F1: {best_f1:.4f}  (P={best_prec:.4f}, R={best_rec:.4f})")

        # Metrics at best threshold
        final_preds = (probs > best_thresh).astype(int)
        print(f"   Accuracy@thr: {((final_preds==true).mean()):.4f}")
        print(f"   Precision@thr: {precision_score(true, final_preds, zero_division=0):.4f}")
        print(f"   Recall@thr:    {recall_score   (true, final_preds, zero_division=0):.4f}")
        print(f"   F1@thr:        {f1_score(true, final_preds):.4f}")

