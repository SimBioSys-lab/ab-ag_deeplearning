import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel
from torch.amp import autocast
import csv
import os
import sys
import shap
import random

# -----------------------------
# Configuration & helpers
# -----------------------------
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

inp_file = sys.argv[1]  # a text file with model paths, one per line

# Test configuration
test_config = {
    'batch_size': 1,
    'sequence_file': 'para_test_esmsequences_1600.npz',
    'data_file': 'para_test_esminterfaces_1600.npz',
    'edge_file': 'para_test_esmedges_1600.npz',
    'max_len': 1600,
    'vocab_size': 31,
    'num_classes': 2,
    'pad_token_id': 0,    # adjust if your PAD differs
    'eoc_token_id': 24    # adjust to your EOC id (you used 24 below)
}

# Collate (unchanged)
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

# Dataset/DataLoader
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Interfaces keys (to align outputs & per-sample SHAP saving)
interfaces_npz = np.load(test_config['data_file'], allow_pickle=True)
interface_keys = list(interfaces_npz.keys())

# ------------------------------------------
# SHAP: model wrapper returning a scalar
# ------------------------------------------
class ScalarProbWrapper(torch.nn.Module):
    """
    Wraps ClassificationModel to return a scalar per-sample:
      mean positive-class probability over VALID positions.
    We define "valid" as pt >= 0 (your mask convention).
    You can customize to antibody-only / antigen-only by editing the mask region.
    """
    def __init__(self, base_model, eoc_token_id=24):
        super().__init__()
        self.base = base_model
        self.eoc_token_id = eoc_token_id

    def forward(self, inputs):
        """
        inputs is a tuple (sequences, padded_edges, pt_tensor)
        sequences: [B, C, L] integer-like (C includes the sequence channel at index 0)
        padded_edges: [B, 2, E]
        pt_tensor: [B, L] with >=0 for valid positions
        Returns: [B,] scalar mean positive-class probability
        """
        sequences, padded_edges, pt_tensor = inputs
        # We don't need autocast here for SHAP stability
        outputs = self.base(sequences=sequences, padded_edges=padded_edges, return_attention=False)[0]
        # outputs: [B, L, num_classes]; take prob of class 1
        probs = torch.softmax(outputs, dim=-1)[..., 1]  # [B, L]
        # Valid mask (edit here if you want only H/L or only antigens)
        valid_mask = (pt_tensor >= 0)  # [B, L]
        # Fallback if mask is empty (avoid NaN)
        denom = torch.clamp(valid_mask.sum(dim=1, keepdim=True), min=1)
        mean_pos = (probs * valid_mask.float()).sum(dim=1) / denom.squeeze(1)
        return mean_pos  # [B]

# -----------------------------
# Utility: build SHAP background
# -----------------------------
def build_background(loader, k=8):
    """
    Collect a small background set (k samples) from the test loader.
    Returns three tensors concatenated along batch dim:
      bg_sequences: [k, C, L], bg_edges: [k, 2, E], bg_pt: [k, L]
    """
    bg_seq, bg_edges, bg_pt = [], [], []
    taken = 0
    with torch.no_grad():
        for padded_edges, sequence_tensor, pt_tensor in loader:
            bg_seq.append(sequence_tensor)
            bg_edges.append(padded_edges)
            bg_pt.append(pt_tensor)
            taken += 1
            if taken >= k:
                break
    bg_sequences = torch.cat(bg_seq, dim=0)
    bg_edges = torch.cat(bg_edges, dim=0)
    bg_pt = torch.cat(bg_pt, dim=0)
    return bg_sequences, bg_edges, bg_pt

# -----------------------------
# Read model list
# -----------------------------
with open(inp_file, 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# -----------------------------
# Main loop: test & SHAP
# -----------------------------
for model_file in models:
    torch.cuda.empty_cache()
    print(f"\n=== Testing & Explaining model: {model_file}")

    # Parse your hyperparams from filename
    parts = model_file.split('_')
    num_layers = int(parts[1][1:])
    num_gnn_layers = int(parts[2][1:])
    num_int_layers = int(parts[3][1:])
    embed_dim = 256
    num_heads = int(parts[7][5:])
    dropout = float(parts[4][2:])
    drop_path_rate = float(parts[5][3:])

    # Build base model
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

    # Load checkpoint (strip SWA, dp prefixes)
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    checkpoint.pop("n_averaged", None)
    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print("Missing keys :", missing)
    if unexpected:
        print("Unexpected  :", unexpected)
    model.eval()

    # -----------------------------
    # Metrics accumulation
    # -----------------------------
    metrics = {
        "Lchain": {"true": [], "pred": [], "probs": []},
        "Hchain": {"true": [], "pred": [], "probs": []},
        "Antibody": {"true": [], "pred": [], "probs": []},
        "Antigen": {"true": [], "pred": [], "probs": []}
    }

    # -----------------------------
    # Prepare SHAP explainer
    # -----------------------------
    # Build a small background from the SAME loader (first k samples)
    bg_seq_cpu, bg_edges_cpu, bg_pt_cpu = build_background(test_loader, k=8)
    bg_seq = bg_seq_cpu.to(device, non_blocking=True)
    bg_edges = bg_edges_cpu.to(device, non_blocking=True)
    bg_pt = bg_pt_cpu.to(device, non_blocking=True)

    # Wrap model to scalar output
    wrapped = ScalarProbWrapper(model, eoc_token_id=test_config['eoc_token_id']).to(device)
    wrapped.eval()

    # SHAP DeepExplainer (gradient-based, fast for torch)
    # Pass background as a tuple of tensors
    explainer = shap.DeepExplainer(wrapped, (bg_seq, bg_edges, bg_pt))

    # Files for outputs
    pred_csv_path = f"{model_file}_predictions.csv"
    shap_npz_path = f"{model_file}_shap_token_values.npz"
    shap_topk_csv = f"{model_file}_shap_topk_tokens.csv"

    # We'll store per-sample SHAP over the sequence channel (token importance over positions)
    shap_dict = {}  # key: interface_keys[i] -> np.array([L]) SHAP values

    with open(pred_csv_path, "w", newline="") as fcsv, open(shap_topk_csv, "w", newline="") as ftop:
        writer = csv.writer(fcsv)
        writer.writerow(["Chain Type", "True Label", "Predicted Label", "Probability"])

        topw = csv.writer(ftop)
        topw.writerow(["sample_key", "topk_positions", "topk_SHAP_values"])

        sample_idx = 0
        with torch.no_grad():
            for padded_edges, sequence_tensor, pt_tensor in test_loader:
                # Move to device
                padded_edges = padded_edges.to(device, non_blocking=True)
                sequence_tensor = sequence_tensor.to(device, non_blocking=True)
                pt_tensor = pt_tensor.to(device, non_blocking=True)

                # Forward for predictions
                with autocast("cuda"):
                    outputs, _ = model(sequences=sequence_tensor, padded_edges=padded_edges, return_attention=True)
                outputs = outputs.squeeze(1)                  # [B=1, L, C]
                probs = torch.softmax(outputs, dim=-1)        # [1, L, 2]
                preds = torch.argmax(probs, dim=-1)           # [1, L]

                # EOCs from the sequence channel 0
                seq_ids = sequence_tensor[0, 0, :].detach().cpu().numpy()
                EOCs = np.where(seq_ids == test_config['eoc_token_id'])[0]

                # Define chain segments
                chain_types = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(EOCs) - 2)]
                chain_splits = [0] + EOCs.tolist()

                # Ensure metrics dict has AG entries
                for chain_type in chain_types[2:]:
                    if chain_type not in metrics:
                        metrics[chain_type] = {"true": [], "pred": [], "probs": []}

                # Update per-chain metrics & write out predictions
                for i, chain_type in enumerate(chain_types):
                    start, end = chain_splits[i], chain_splits[i + 1]
                    mask = (pt_tensor[0, start:end] >= 0)

                    if mask.sum().item() == 0:
                        print(f"No valid positions for {chain_type}. Skipping...")
                        continue

                    true_labels = pt_tensor[0, start:end][mask].detach().cpu().numpy()
                    predicted_labels = preds[0, start:end][mask].detach().cpu().numpy()
                    probabilities = probs[0, start:end, 1][mask].detach().cpu().numpy()

                    metrics[chain_type]["true"].extend(true_labels)
                    metrics[chain_type]["pred"].extend(predicted_labels)
                    metrics[chain_type]["probs"].extend(probabilities)

                    # Group into antibody/antigen buckets
                    if chain_type in ["Lchain", "Hchain"]:
                        metrics["Antibody"]["true"].extend(true_labels)
                        metrics["Antibody"]["pred"].extend(predicted_labels)
                        metrics["Antibody"]["probs"].extend(probabilities)
                    else:
                        metrics["Antigen"]["true"].extend(true_labels)
                        metrics["Antigen"]["pred"].extend(predicted_labels)
                        metrics["Antigen"]["probs"].extend(probabilities)

                    # Write per-position predictions
                    for t, p, pr in zip(true_labels, predicted_labels, probabilities):
                        writer.writerow([chain_type, int(t), int(p), float(pr)])

                # -----------------------------
                # SHAP for this sample
                # -----------------------------
                # NOTE: SHAP will attribute importance over the *entire* input.
                # We will extract token-level attributions from the sequence channel only:
                # sequence_tensor shape: [1, C, L]. We'll take channel 0 (token ids proxy).
                # DeepExplainer supports tuple inputs that match the wrapped forward signature.
                # It returns a tuple of SHAP arrays with the same shapes as inputs.
                shap_vals_tuple = explainer.shap_values((sequence_tensor, padded_edges, pt_tensor))
                # shap_vals_tuple is a list/tuple with 3 elements (matching inputs)
                shap_seq = shap_vals_tuple[0]  # [1, C, L]
                # Aggregate over channels except the token channel; keep channel 0
                # If your "sequence channel" is strictly integer token ids, SHAP on that channel
                # reflects attribution to those discrete IDs as used by your embedding.
                # We take channel 0 directly:
                token_shap = shap_seq[0, 0, :].detach().cpu().numpy()  # [L]

                # Save SHAP vector for this sample
                sample_key = interface_keys[sample_idx] if sample_idx < len(interface_keys) else f"sample_{sample_idx}"
                shap_dict[sample_key] = token_shap

                # Also write a quick top-k summary
                k = 10
                top_idx = np.argsort(-np.abs(token_shap))[:k].tolist()
                top_vals = [float(token_shap[i]) for i in top_idx]
                topw.writerow([sample_key, top_idx, top_vals])

                sample_idx += 1

    # After all samples for this model: dump SHAP arrays
    np.savez_compressed(shap_npz_path, **shap_dict)
    print(f"Saved SHAP token attributions to {shap_npz_path}")
    print(f"Saved prediction CSV to {pred_csv_path}")
    print(f"Saved SHAP top-k CSV to {shap_topk_csv}")

    # -----------------------------
    # Print final metrics
    # -----------------------------
    for chain_type, data in metrics.items():
        true = data["true"]; pred = data["pred"]; probs = data["probs"]
        if len(true) == 0:
            print(f"No data available for {chain_type}")
            continue
        accuracy = np.mean(np.array(true) == np.array(pred))
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        auc_roc = roc_auc_score(true, probs)
        precision_vals, recall_vals, _ = precision_recall_curve(true, probs)
        auc_pr = auc(recall_vals, precision_vals)
        print(f"\nMetrics for {chain_type}:")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")

