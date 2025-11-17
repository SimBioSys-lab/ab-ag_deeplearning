#!/usr/bin/env python3
import os
import csv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from torch.amp import autocast
from collections import defaultdict

from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel

# =========================
# Config & CuDNN settings
# =========================
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

CSV_FLOAT_FMT = '%.6g'  # compact float printing for CSVs

# =========================
# Read model filenames
# =========================
with open('ihd10_newesm_i_files', 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# =========================
# Test configuration
# =========================
test_config = {
    'batch_size': 1,
    'sequence_file': 'ihd_test_esmsequences_1600.npz',
    'data_file': 'ihd_test_esminterfaces10_1600.npz',
    'edge_file': 'ihd_test_esmedges_1600.npz',
    'max_len': 1600,
    'vocab_size': 31,
    'num_classes': 2
}

# =========================
# Collate
# =========================
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
    padded_edges = torch.stack(padded_edges)
    return padded_edges, sequence_tensor, pt_tensor

# =========================
# Dataset & Loader
# =========================
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

# =========================
# Device
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Interface keys for naming
# =========================
interfaces_npz = np.load(test_config['data_file'], allow_pickle=True)
interface_keys = list(interfaces_npz.keys())

# =========================
# Try to import DyM class (optional)
# =========================
try:
    from Models_fullnew import DyM as DyMClass
except Exception:
    DyMClass = None
assert DyMClass is not None, "DyM not imported; you’re saving a dummy mask=1"
# =========================
# Helpers to save DyM & attention to CSV
# =========================
def save_dym_to_csv(mask_t: torch.Tensor, out_csv: str):
    """
    mask_t: torch.Tensor with shape [B, L, 1] or [B, L] (batch size is 1 here)
    Writes: CSV with columns: token_idx, mask
    """
    m = mask_t.detach().cpu().numpy()
    m = np.squeeze(m)  # -> [L]
    if m.ndim != 1:
        m = m.reshape(-1)
    df = pd.DataFrame({"token_idx": np.arange(m.shape[0]), "mask": m})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, float_format=CSV_FLOAT_FMT)

def save_attn_to_csv(attn_t: torch.Tensor, out_prefix: str):
    """
    attn_t: [B, H, L, L] or [H, L, L]
    Writes: one CSV per head: <out_prefix>__head{h}.csv
    """
    a = attn_t.detach().cpu().float().numpy()
    if a.ndim == 4:
        a = a[0]  # take batch 0 -> [H, L, L]
    elif a.ndim != 3:
        raise ValueError(f"Unexpected attention ndim={a.ndim} (expected 3 or 4).")

    H = a.shape[0]
    for h in range(H):
        out_csv = f"{out_prefix}__head{h}.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        np.savetxt(out_csv, a[h], delimiter=",", fmt=CSV_FLOAT_FMT)

# =========================
# Hook: capture DyM mask per forward
# We infer mask ≈ mean over channels of (output / input)
# =========================
dym_batch = defaultdict(list)  # cleared per batch

def dym_hook(module, inputs, output):
    x = inputs[0]
    mask_est = (output / (x + 1e-8)).mean(dim=-1, keepdim=True).detach().cpu()
    lname = getattr(module, "_layer_name", module.__class__.__name__)
    dym_batch[lname].append(mask_est)

# =========================
# Test each model
# =========================
for model_file in models:
    torch.cuda.empty_cache()
    print(f"\nTesting model: {model_file}")

    # ----- Parse model config from filename -----
    parts = model_file.split('_')
    num_layers = int(parts[1][1:])
    num_gnn_layers = int(parts[2][1:])
    num_int_layers = int(parts[3][1:])
    embed_dim = 256
    num_heads = 16
    dropout = float(parts[4][2:])
    drop_path_rate = float(parts[5][3:])

    # ----- Init model -----
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

    # ----- Load checkpoint -----
    try:
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_file, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint.pop("n_averaged", None)  # from SWA
    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print("Missing keys :", missing)
    if unexpected:
        print("Unexpected  :", unexpected)
    model.eval()

    # ----- Register DyM hooks -----
    n_hooks = 0
    for name, module in model.named_modules():
        is_dym = (DyMClass is not None and isinstance(module, DyMClass)) or (module.__class__.__name__ == "DyM")
        if is_dym:
            module._layer_name = name  # for file naming
            module.register_forward_hook(dym_hook)
            n_hooks += 1
    print(f"[HOOK] Registered on {n_hooks} DyM modules")

    # ----- Output dirs for this model -----
    base_name = os.path.splitext(os.path.basename(model_file))[0]
    dym_dir = f"{base_name}_dym_masks_csv"
    attn_dir = f"{base_name}_attn_csv"
    os.makedirs(dym_dir, exist_ok=True)
    os.makedirs(attn_dir, exist_ok=True)

    # ----- Metrics accumulators -----
    metrics = {
        "Lchain": {"true": [], "pred": [], "probs": []},
        "Hchain": {"true": [], "pred": [], "probs": []},
        "Antibody": {"true": [], "pred": [], "probs": []},
        "Antigen": {"true": [], "pred": [], "probs": []}
    }

    # ----- CSV for predictions -----
    with open(f"{model_file}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Chain Type", "True Label", "Predicted Label", "Probability"])

        sample_idx = 0
        with torch.no_grad():
            for padded_edges, sequence_tensor, pt_tensor in test_loader:
                # move to device
                padded_edges = padded_edges.to(device)
                sequence_tensor = sequence_tensor.to(device)
                pt_tensor = pt_tensor.to(device)

                # forward
                with autocast("cuda"):
                    outputs, last_attention = model(
                        sequences=sequence_tensor,
                        padded_edges=padded_edges,
                        return_attention=True
                    )

                # -------- Save attention (CSV per head) --------
                sample_key = interface_keys[sample_idx] if sample_idx < len(interface_keys) else f"sample_{sample_idx}"
                attn_prefix = os.path.join(attn_dir, sample_key)
                save_attn_to_csv(last_attention, attn_prefix)

                # -------- Save DyM masks from this batch (CSV per DyM layer) --------
                for layer_name, masks in dym_batch.items():
                    if len(masks) == 0:
                        continue
                    out_csv = os.path.join(dym_dir, f"{layer_name}__{sample_key}.csv")
                    save_dym_to_csv(masks[0], out_csv)
                dym_batch.clear()  # IMPORTANT

                # predictions
                outputs = outputs.squeeze(1)
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                # chain splits (EOC token id == 24)
                EOCs = np.where(sequence_tensor[0, 0, :].detach().cpu().numpy() == 24)[0]
                chain_types = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(EOCs) - 2)]
                chain_splits = [0] + EOCs.tolist()

                # ensure metrics dict has AG entries
                for chain_type in chain_types[2:]:
                    if chain_type not in metrics:
                        metrics[chain_type] = {"true": [], "pred": [], "probs": []}

                # per-chain metrics
                for i, chain_type in enumerate(chain_types):
                    start, end = chain_splits[i], chain_splits[i + 1]
                    mask = pt_tensor[0, start:end] >= 0
                    if mask.sum().item() == 0:
                        print(f"No valid positions for {chain_type}. Skipping...")
                        continue

                    true_labels = pt_tensor[0, start:end][mask].detach().cpu().numpy()
                    predicted_labels = preds[0, start:end][mask].detach().cpu().numpy()
                    probabilities = probs[0, start:end, 1][mask].detach().cpu().numpy()
                    metrics[chain_type]["true"].extend(true_labels)
                    metrics[chain_type]["pred"].extend(predicted_labels)
                    metrics[chain_type]["probs"].extend(probabilities)

                    if chain_type in ["Lchain", "Hchain"]:
                        metrics["Antibody"]["true"].extend(true_labels)
                        metrics["Antibody"]["pred"].extend(predicted_labels)
                        metrics["Antibody"]["probs"].extend(probabilities)
                    else:
                        metrics["Antigen"]["true"].extend(true_labels)
                        metrics["Antigen"]["pred"].extend(predicted_labels)
                        metrics["Antigen"]["probs"].extend(probabilities)

                    for t, p, pr in zip(true_labels, predicted_labels, probabilities):
                        writer.writerow([chain_type, t, p, pr])

                sample_idx += 1  # next sample

    # ----- Final metrics -----
    for chain_type, data in metrics.items():
        true = data["true"]
        pred = data["pred"]
        probs = data["probs"]

        if len(true) == 0:
            print(f"No data available for {chain_type}")
            continue

        accuracy = np.mean(np.array(true) == np.array(pred))
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        auc_roc = roc_auc_score(true, probs) if len(np.unique(true)) > 1 else float('nan')
        precision_vals, recall_vals, _ = precision_recall_curve(true, probs)
        auc_pr = auc(recall_vals, precision_vals) if len(np.unique(true)) > 1 else float('nan')

        print(f"\nMetrics for {chain_type}:")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")

