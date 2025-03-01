import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_all import SequenceParatopeDataset
from Models_new import ClassificationModel
from torch.cuda.amp import autocast  # For mixed precision inference
import csv

# Read model filenames
with open('model_g_files', 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# Configuration for the test
test_config = {
    'batch_size': 1,  # Reduced batch size to manage memory
    'sequence_file': 'preprocessed_test_sequences.npz',
    'data_file': 'test_data.npz',
    'edge_file': 'test_edge_lists.npz',
    'max_len': 1200,
    'vocab_size': 22,
    'num_classes': 2
}


def custom_collate_fn(batch):
    """
    Custom collate function to pad edge indices dynamically based on the longest edge index in the batch.
    """
    sequences, ss, sasa, pt, sapt, edge = zip(*batch)

    # Stack sequences into a single tensor
    sequence_tensor = torch.stack(sequences)  # Shape: [batch_size, 100, seq_len, feature_dim]

    # Convert ss, sasa, pt, and sapt to tensors
    ss_tensor = torch.tensor(np.array(ss), dtype=torch.long)
    sasa_tensor = torch.tensor(np.array(sasa), dtype=torch.float)
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long)
    sapt_tensor = torch.tensor(np.array(sapt), dtype=torch.float)

    # Determine the maximum number of edges in the batch
    max_edges = max(edge_index.shape[0] for edge_index in edge)

    # Pad edge indices dynamically
    padded_edges = []
    for edge_index in edge:
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)  # Initialize with -1
        edge_pad[:, :edge_index.shape[0]] = torch.tensor(edge_index.T, dtype=torch.long)  # Fill valid edges
        padded_edges.append(edge_pad)

    # Stack padded edges
    padded_edges = torch.stack(padded_edges)  # Shape: [batch_size, 2, max_edges]

    return padded_edges, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor

# Load the test dataset
test_dataset = SequenceParatopeDataset(data_file=test_config['data_file'], sequence_file=test_config['sequence_file'], edge_file=test_config['edge_file'], max_len=test_config['max_len'])
test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'], collate_fn=custom_collate_fn, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test each model
for model_file in models:
    print(f"\nTesting model: {model_file}")

    # Extract model parameters from the filename
    parts = model_file.split('_')
    num_layers = int(parts[1][1:])
    num_gnn_layers = int(parts[2][1:])
    embed_dim = int(parts[3][1:])
    num_heads = int(parts[4][1:])
    accum_steps = int(parts[5].split('.')[0][-1])  # Strip ".pth"

    # Update test configuration
    test_config['embed_dim'] = embed_dim
    test_config['num_heads'] = num_heads
    test_config['num_layers'] = num_layers
    test_config['num_gnn_layers'] = num_gnn_layers
    # Initialize the model
    model = ClassificationModel(
        vocab_size=test_config['vocab_size'],
        seq_len=test_config['max_len'],
        embed_dim=test_config['embed_dim'],
        num_heads=test_config['num_heads'],
        num_layers=test_config['num_layers'],
        num_gnn_layers=test_config['num_gnn_layers'],
        num_classes=test_config['num_classes']
    )
    model = model.to(device)

    # Load model weights
    checkpoint = torch.load(model_file, map_location=device)
    if 'module.' in list(checkpoint.keys())[0]:
        checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}  # Remove 'module.' prefix
    model.load_state_dict(checkpoint)
    model.eval()

    # Initialize metrics and predictions
    correct, total = 0, 0
    flattened_probs, flattened_targets, flattened_predictions = [], [], []

    # Open CSV for saving intermediate results
    csv_filename = f"{model_file}_predictions.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True Label", "Predicted Label", "Probability"])  # CSV header

        # Testing loop
        with torch.no_grad():
            for padded_edges, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor in test_loader:
                sequence_tensor, pt_tensor, padded_edges = sequence_tensor.to(device), pt_tensor.to(device), padded_edges.to(device)

                # Use mixed precision for inference
                with autocast():
                    outputs = model(sequence_tensor, padded_edges)
                probs = torch.softmax(outputs, dim=2)
                preds = torch.argmax(probs, dim=2)

                # Mask valid positions (ignore padding)
                mask = pt_tensor >= 0
                correct += (preds[mask] == pt_tensor[mask]).sum().item()
                total += mask.sum().item()

                # Save metrics and predictions in chunks
                for i in range(sequence_tensor.size(0)):  # Iterate over the batch
                    true_labels = pt_tensor[i][mask[i]].cpu().numpy()
                    predicted_labels = preds[i][mask[i]].cpu().numpy()
                    probabilities = probs[i, :, 1][mask[i]].cpu().numpy()

                    flattened_targets.extend(true_labels)
                    flattened_predictions.extend(predicted_labels)
                    flattened_probs.extend(probabilities)

                    for t, p, pr in zip(true_labels, predicted_labels, probabilities):
                        writer.writerow([t, p, pr])

    # Compute final metrics
    accuracy = correct / total if total > 0 else 0
    precision = precision_score(flattened_targets, flattened_predictions)
    recall = recall_score(flattened_targets, flattened_predictions)
    auc_roc = roc_auc_score(flattened_targets, flattened_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(flattened_targets, flattened_probs)
    auc_pr = auc(recall_vals, precision_vals)

    # Print metrics
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  AUC-PR: {auc_pr:.4f}")

    # Save metrics to a file
    metrics_filename = f"{model_file}_metrics.txt"
    with open(metrics_filename, "w") as metric_file:
        metric_file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        metric_file.write(f"Precision: {precision:.4f}\n")
        metric_file.write(f"Recall: {recall:.4f}\n")
        metric_file.write(f"AUC-ROC: {auc_roc:.4f}\n")
        metric_file.write(f"AUC-PR: {auc_pr:.4f}\n")

    print(f"Results saved to {csv_filename} and {metrics_filename}")

