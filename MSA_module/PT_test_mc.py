import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_new import ClassificationModel
from torch.amp import autocast
import csv
import os

# Configuration
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Read model filenames
with open('mcmodel_files', 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# Test configuration
test_config = {
    'batch_size': 1,
    'sequence_file': 'padded_sequences_test_3000.npz',
    'data_file': 'padded_interfaces_test_3000.npz',
    'edge_file': 'padded_edges_test_3000.npz',
    'max_len': 3000,
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
test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'], collate_fn=custom_collate_fn, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    num_heads = 16
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

    # Initialize metrics for each chain and the combined antibody and antigen
    metrics = {
        "Lchain": {"true": [], "pred": [], "probs": []},
        "Hchain": {"true": [], "pred": [], "probs": []},
        "Antibody": {"true": [], "pred": [], "probs": []},
        "Antigen": {"true": [], "pred": [], "probs": []}
    }

    with open(f"{model_file}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Chain Type", "True Label", "Predicted Label", "Probability"])

        with torch.no_grad():
            for padded_edges, sequence_tensor, pt_tensor in test_loader:
                sequence_tensor, pt_tensor, padded_edges = (
                    sequence_tensor.to(device),
                    pt_tensor.to(device),
                    padded_edges.to(device),
                )

                # Perform inference
                with autocast("cuda"):
                    outputs = model(sequence_tensor, padded_edges)
                outputs = outputs.squeeze(1)
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                # Extract EOCs
                EOCs = np.where(sequence_tensor[0, 0, :].cpu().numpy() == 22)[0]

                # Define chain types and splits
                chain_types = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(EOCs) - 2)]
                chain_splits = [0] + EOCs.tolist()

                # Ensure metrics are initialized for AGchains
                for chain_type in chain_types[2:]:
                    if chain_type not in metrics:
                        metrics[chain_type] = {"true": [], "pred": [], "probs": []}

                # Process metrics for each chain
                for i, chain_type in enumerate(chain_types):
                    start, end = chain_splits[i], chain_splits[i + 1]
                    mask = pt_tensor[0, start:end] >= 0

                    if mask.sum().item() == 0:
                        print(f"No valid positions for {chain_type}. Skipping...")
                        continue
                    true_labels = pt_tensor[0, start:end][mask].cpu().numpy()
                    predicted_labels = preds[0, start:end][mask].cpu().numpy()
                    probabilities = probs[0, start:end, 1][mask].cpu().numpy()

                    # Log metrics
                    metrics[chain_type]["true"].extend(true_labels)
                    metrics[chain_type]["pred"].extend(predicted_labels)
                    metrics[chain_type]["probs"].extend(probabilities)

                    # Combine Antibody and Antigen metrics
                    if chain_type in ["Lchain", "Hchain"]:
                        metrics["Antibody"]["true"].extend(true_labels)
                        metrics["Antibody"]["pred"].extend(predicted_labels)
                        metrics["Antibody"]["probs"].extend(probabilities)
                    else:
                        metrics["Antigen"]["true"].extend(true_labels)
                        metrics["Antigen"]["pred"].extend(predicted_labels)
                        metrics["Antigen"]["probs"].extend(probabilities)

                    # Write predictions to CSV
                    for t, p, pr in zip(true_labels, predicted_labels, probabilities):
                        writer.writerow([chain_type, t, p, pr])

    # Calculate and print final metrics
    for chain_type, data in metrics.items():
        true = data["true"]
        pred = data["pred"]
        probs = data["probs"]

        if len(true) == 0:
            print(f"No data available for {chain_type}")
            continue

        accuracy = np.mean(np.array(true) == np.array(pred))
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        auc_roc = roc_auc_score(true, probs)
        precision_vals, recall_vals, _ = precision_recall_curve(true, probs)
        auc_pr = auc(recall_vals, precision_vals)

        print(f"\nMetrics for {chain_type}:")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")

