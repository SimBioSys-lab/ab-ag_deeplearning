import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_mask import ClassificationModel
from torch.amp import autocast
import csv
import os

# Configuration
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Read model filenames
with open('comb45_files', 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# Test configuration
test_config = {
    'batch_size': 1,
    'sequence_file': 'combined_np_test_sequences_1600.npz',
    'data_file': 'combined_np_test_interface4.5_1600.npz',
    'edge_file': 'combined_np_test_edges_1600.npz',
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
    padded_edges = torch.stack(padded_edges)
    return padded_edges, sequence_tensor, pt_tensor

# Load the dataset
test_dataset = SequenceParatopeDataset(
    data_file=test_config['data_file'],
    sequence_file=test_config['sequence_file'],
    edge_file=test_config['edge_file'],
    max_len=test_config['max_len']
)
test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'],
                           collate_fn=custom_collate_fn, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the interfaces NPZ file and get its keys.
interfaces_npz = np.load(test_config['data_file'], allow_pickle=True)
interface_keys = list(interfaces_npz.keys())

# Function to find the best threshold for maximizing AUC-PR
def find_best_threshold(true_labels, probabilities):
    precision_vals, recall_vals, thresholds = precision_recall_curve(true_labels, probabilities)
    
    # Ensure recall values are strictly increasing
    sorted_indices = np.argsort(recall_vals)
    recall_vals = recall_vals[sorted_indices]
    precision_vals = precision_vals[sorted_indices]

    # Compute AUC-PR for each threshold
    auc_pr_values = [auc(recall_vals[:i+1], precision_vals[:i+1]) for i in range(len(thresholds))]

    # Find the threshold that maximizes AUC-PR
    best_threshold_idx = np.argmax(auc_pr_values)
    best_threshold = thresholds[best_threshold_idx]
    best_auc_pr = auc_pr_values[best_threshold_idx]
    
    return best_threshold, best_auc_pr
# Test each model
for model_file in models:
    torch.cuda.empty_cache()
    print(f"\nTesting model: {model_file}")

    # Parse model configuration from filename
    parts = model_file.split('_')
    num_layers = int(parts[2][1])
    num_gnn_layers = int(parts[3][1])
    num_int_layers = int(parts[4][1])
    embed_dim = 256
    num_heads = 8
    dropout = float(parts[5][2:5])

    # Initialize model
    model = ClassificationModel(
        vocab_size=test_config['vocab_size'],
        seq_len=test_config['max_len'],
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers,
        num_gnn_layers=num_gnn_layers,
        num_int_layers=num_int_layers,
        num_classes=test_config['num_classes']
    )
    model = model.to(device)

    # Load model weights
    checkpoint = torch.load(model_file, map_location=device)
    if 'module.' in list(checkpoint.keys())[0]:
        checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()

    # Initialize metrics for each chain
    metrics = {
        "Lchain": {"true": [], "pred": [], "probs": []},
        "Hchain": {"true": [], "pred": [], "probs": []},
        "Antibody": {"true": [], "pred": [], "probs": []},
        "Antigen": {"true": [], "pred": [], "probs": []}
    }

    # Prepare to save predictions to CSV
    with open(f"{model_file}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Chain Type", "True Label", "Predicted Label", "Probability"])

        with torch.no_grad():
            for padded_edges, sequence_tensor, pt_tensor in test_loader:
                # Move tensors to device
                padded_edges = padded_edges.to(device)
                sequence_tensor = sequence_tensor.to(device)
                pt_tensor = pt_tensor.to(device)

                # Perform inference
                with autocast("cuda"):
                    outputs, _ = model(sequences=sequence_tensor, padded_edges=padded_edges, return_attention=True)

                outputs = outputs.squeeze(1)
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                # Define chain types and splits
                EOCs = np.where(sequence_tensor[0, 0, :].cpu().numpy() == 22)[0]
                chain_types = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(EOCs) - 2)]
                chain_splits = [0] + EOCs.tolist()

                for chain_type in chain_types[2:]:
                    if chain_type not in metrics:
                        metrics[chain_type] = {"true": [], "pred": [], "probs": []}

                # Process metrics for each chain type
                for i, chain_type in enumerate(chain_types):
                    start, end = chain_splits[i], chain_splits[i + 1]
                    mask = pt_tensor[0, start:end] >= 0

                    if mask.sum().item() == 0:
                        continue
                    true_labels = pt_tensor[0, start:end][mask].cpu().numpy()
                    predicted_labels = preds[0, start:end][mask].cpu().numpy()
                    probabilities = probs[0, start:end, 1][mask].cpu().numpy()

                    # Log metrics
                    metrics[chain_type]["true"].extend(true_labels)
                    metrics[chain_type]["pred"].extend(predicted_labels)
                    metrics[chain_type]["probs"].extend(probabilities)

                    # Aggregate for Antibody/Antigen
                    if chain_type in ["Lchain", "Hchain"]:
                        metrics["Antibody"]["true"].extend(true_labels)
                        metrics["Antibody"]["pred"].extend(predicted_labels)
                        metrics["Antibody"]["probs"].extend(probabilities)
                    else:
                        metrics["Antigen"]["true"].extend(true_labels)
                        metrics["Antigen"]["pred"].extend(predicted_labels)
                        metrics["Antigen"]["probs"].extend(probabilities)

    # Calculate final metrics
    for chain_type, data in metrics.items():
        true = np.array(data["true"])
        probs = np.array(data["probs"])

        if len(true) == 0:
            continue

        auc_pr = auc(*precision_recall_curve(true, probs)[:2])

        # Find best threshold for Antibody
        if chain_type == "Antibody":
            best_threshold, best_auc_pr = find_best_threshold(true, probs)
            print(f"  Best Threshold for {chain_type}: {best_threshold:.4f} (AUC-PR: {best_auc_pr:.4f})")

        print(f"\nMetrics for {chain_type}: AUC-PR: {auc_pr:.4f}")

