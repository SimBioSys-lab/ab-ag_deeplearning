import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_ctm import SequenceParatopeDataset
from Models_test import CTMModel
from torch.amp import autocast
import os

# Configuration
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Read model filenames
with open('ctm_files', 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# Test configuration
test_config = {
    'batch_size': 1,
    'sequence_file': 'padded_test_sequences_2400.npz',
    'data_file': 'global_maps_test_aligned_2400.npz',
    'edge_file': 'padded_test_edges_2400.npz',
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
    model = CTMModel(
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
    
    # Dictionaries to store predictions and last_attention matrices for each sample.
    preds_dict = {}
    last_attention_dict = {}
    sample_idx = 0
    
    with torch.no_grad():
        for padded_edges, sequence_tensor, pt_tensor in test_loader:
            padded_edges = padded_edges.to(device)
            sequence_tensor = sequence_tensor.to(device)
            pt_tensor = pt_tensor.to(device)
            
            with autocast("cuda"):
                outputs, last_attention = model(sequences=sequence_tensor,
                                                padded_edges=padded_edges,
                                                return_attention=True,
                                                tied=False)
            
            # Compute probabilities over the channel dimension.
            probs = torch.softmax(outputs, dim=1)
            # Argmax over channel dim gives predictions; shape: [batch_size, seq_len, seq_len]
            preds = torch.argmax(probs, dim=1)
            
            # Use interface_keys if available; otherwise, use a default key.
            key = interface_keys[sample_idx] if sample_idx < len(interface_keys) else f"sample_{sample_idx}"
            preds_dict[key] = preds.cpu().numpy()
            last_attention_dict[key] = last_attention.cpu().numpy()
            sample_idx += 1
    
    # Save the predictions and last_attention matrices to NPZ files.
    np.savez(f"{model_file}_preds.npz", **preds_dict)
    np.savez(f"{model_file}_last_attention.npz", **last_attention_dict)
    print(f"Saved predictions to {model_file}_preds.npz")
    print(f"Saved last_attention matrices to {model_file}_last_attention.npz")

