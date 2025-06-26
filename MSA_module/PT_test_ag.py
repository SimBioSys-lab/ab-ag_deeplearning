import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_test import ClassificationModel
from torch.amp import autocast
import csv
import os

# Configuration
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Read model filenames
with open('mcagmodel_files', 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# Test configuration
test_config = {
    'batch_size': 1,
    'sequence_file': 'antigen_test_sequences_2400.npz',
    'data_file': 'antigen_test_interfaces_2400.npz',
    'edge_file': 'antigen_test_edges_2400.npz',
    'max_len': 2400,
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
    num_heads = 4
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

    # Initialize overall metrics dictionary
    metrics = {
        "Overall": {"true": [], "pred": [], "probs": []}
    }
    
    # Prepare to save predictions to CSV
    with open(f"{model_file}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "True Label", "Predicted Label", "Probability"])

        # This dictionary will store the last_attention matrices using keys from the interfaces NPZ.
        last_attention_dict = {}
        sample_idx = 0  # To index into interface_keys

        with torch.no_grad():
            for padded_edges, sequence_tensor, pt_tensor in test_loader:
                # Move tensors to device
                padded_edges = padded_edges.to(device)
                sequence_tensor = sequence_tensor.to(device)
                pt_tensor = pt_tensor.to(device)

                # Perform inference and get last_attention from the model.
                with autocast("cuda"):
                    outputs, last_attention = model(sequences=sequence_tensor, padded_edges=padded_edges,
                                                    return_attention=True)
                # Squeeze channel dimension if necessary (depending on model output shape)
                outputs = outputs.squeeze(1)
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                # Store the last_attention matrix for this sample
                if sample_idx < len(interface_keys):
                    last_attention_dict[interface_keys[sample_idx]] = last_attention.cpu().numpy()
                else:
                    last_attention_dict[f"sample_{sample_idx}"] = last_attention.cpu().numpy()
                sample_idx += 1

                # For overall performance, we use all valid positions in the target.
                # Create a mask for positions where pt_tensor >= 0 (i.e., not padded)
                mask = pt_tensor[0] >= 0
                true_labels = pt_tensor[0][mask].cpu().numpy()
                predicted_labels = preds[0][mask].cpu().numpy()
                # Assuming class 1 probability is of interest
                probabilities = probs[0][mask, 1].cpu().numpy()

                # Aggregate predictions into overall metrics
                metrics["Overall"]["true"].extend(true_labels)
                metrics["Overall"]["pred"].extend(predicted_labels)
                metrics["Overall"]["probs"].extend(probabilities)

                # Write predictions for this sample to CSV
                for t, p, pr in zip(true_labels, predicted_labels, probabilities):
                    writer.writerow([sample_idx, t, p, pr])

    # After processing all batches, calculate and print overall metrics
    overall_true = np.array(metrics["Overall"]["true"])
    overall_pred = np.array(metrics["Overall"]["pred"])
    overall_probs = np.array(metrics["Overall"]["probs"])

    if overall_true.size == 0:
        print("No valid data available for overall evaluation.")
    else:
        accuracy = np.mean(overall_true == overall_pred)
        precision = precision_score(overall_true, overall_pred)
        recall = recall_score(overall_true, overall_pred)
        auc_roc = roc_auc_score(overall_true, overall_probs)
        precision_vals, recall_vals, _ = precision_recall_curve(overall_true, overall_probs)
        auc_pr = auc(recall_vals, precision_vals)

        print("\nOverall Metrics:")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")

    # Save the last_attention matrices to an NPZ file.
    np.savez(f"{model_file}.npz", **last_attention_dict)
    print(f"Saved last_attention matrices to {model_file}.npz")

