import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_SS_SASA import SequenceParatopeDataset
from Models import ParatopeModel

# Configuration for the test
test_config = {
    'batch_size': 4,
    'sequence_file': 'preprocessed_seq_ab_test_1200.npz',
    'pt_file': 'pt_test_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 1,
    'num_classes': 2
}

# Load the test dataset
test_dataset = SequenceParatopeDataset(sequence_file=test_config['sequence_file'], pt_file=test_config['pt_file'], max_len=test_config['seq_len'])
test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False)

# Load the best model state
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParatopeModel(vocab_size=test_config['vocab_size'], seq_len=test_config['seq_len'], embed_dim=test_config['embed_dim'],
                      num_heads=test_config['num_heads'], num_layers=test_config['num_layers'], num_classes=test_config['num_classes'])

# Load state_dict and handle DataParallel prefix if necessary
checkpoint = torch.load('best_paratope_model_with_noise.pth', map_location=device)
if 'module.' in list(checkpoint.keys())[0]:  # Check if keys are prefixed with 'module.'
    checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}  # Remove 'module.' prefix

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# Initialize variables for metrics and predictions
all_probs = []
all_targets = []
all_predictions = []

# Track accuracy
correct = 0
total = 0

# Testing loop
with torch.no_grad():
    for seq, tgt in test_loader:
        seq, tgt = seq.to(device), tgt.to(device)

        # Forward pass
        outputs = model(seq)  # Outputs are logits
        probs = torch.softmax(outputs, dim=2)  # Convert logits to probabilities
        preds = torch.argmax(probs, dim=2)  # Get predicted classes

        # Mask for valid positions (ignore padding)
        mask = tgt >= 0
        correct += (preds[mask] == tgt[mask]).sum().item()
        total += mask.sum().item()

        # Store probabilities, true labels, and predictions for metrics
        all_probs.append(probs.cpu().numpy())  # Probabilities for both classes
        all_targets.append(tgt.cpu().numpy())
        all_predictions.append(preds.cpu().numpy())

# Concatenate all batches
all_probs = np.concatenate(all_probs, axis=0)  # Shape: (num_sequences, seq_len, num_classes)
all_targets = np.concatenate(all_targets, axis=0)  # Shape: (num_sequences, seq_len)
all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (num_sequences, seq_len)

# Calculate metrics
accuracy = correct / total if total > 0 else 0
flattened_mask = all_targets >= 0
flattened_probs = all_probs[:, :, 1][flattened_mask]
flattened_targets = all_targets[flattened_mask]
flattened_predictions = all_predictions[flattened_mask]

precision = precision_score(flattened_targets, flattened_predictions)
recall = recall_score(flattened_targets, flattened_predictions)
auc_roc = roc_auc_score(flattened_targets, flattened_probs)
precision_vals, recall_vals, _ = precision_recall_curve(flattened_targets, flattened_probs)
auc_pr = auc(recall_vals, precision_vals)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")

# Print the first 10 sequences for visual validation
print("\nFirst 10 sequences (True Labels, Predicted Labels, Probabilities):")
for i in range(min(10, len(all_targets))):  # Limit to the first 10 sequences
    true_labels = all_targets[i][all_targets[i] >= 0]  # Ignore padding
    predicted_labels = all_predictions[i][all_targets[i] >= 0]  # Ignore padding
    probabilities = all_probs[i][:len(true_labels), 1]  # Probabilities for the positive class
    print(f"Sequence {i + 1}:")
    print(f"  True Labels: {true_labels.tolist()}")
    print(f"  Predicted Labels: {predicted_labels.tolist()}")
    print(f"  Probabilities: {probabilities.tolist()}")

