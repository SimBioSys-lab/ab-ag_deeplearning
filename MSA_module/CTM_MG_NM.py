import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import os
import numpy as np
from Dataloader_ctm import SequenceParatopeDataset
from Models_NMnew import CTMModel
import torch.nn.functional as F

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Define Focal Loss based on CrossEntropyLoss
class FocalLossCE(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean', ignore_index=-1):
        """
        Focal Loss for multi-class classification.

        Args:
            gamma (float): Focusing parameter.
            weight (Tensor, optional): A manual rescaling weight given to each class.
            reduction (str): 'mean' | 'sum' | 'none'
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(FocalLossCE, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Compute standard cross-entropy loss (per example)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight, ignore_index=self.ignore_index)
        # Compute the probability of the true class for each example
        pt = torch.exp(-ce_loss)
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


config = {
    'batch_size': 4,
    'sequence_file_train': 'cleaned_para_train_sequences_1600.npz',
    'data_file_train': 'global_maps_para_train10.npz',
    'edge_file_train': 'cleaned_para_train_edges_1600.npz',
    'sequence_file_val': 'cleaned_para_val_sequences_1600.npz',
    'data_file_val': 'global_maps_para_val10.npz',
    'edge_file_val': 'cleaned_para_val_edges_1600.npz',
    'max_len': 1600,
    'vocab_size': 23,
    'embed_dim': 256,
    'num_heads': 16,
    'dropout': 0.1,
    'num_layers': 0,
    'num_gnn_layers': 20,
    'num_int_layers': 8,
    'drop_path_rate': 0.1,
    'num_classes': 2,
    'num_epochs': 1000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.1,
    'early_stop_patience': 15,
    'initial_gradient_noise_std': 0.05,
    'accumulation_steps': 2
}
print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# Helper function to add gradient noise
def add_gradient_noise(params, std_dev):
    """Add Gaussian noise to gradients."""
    for param in params:
        if param.grad is not None:
            param.grad.add_(torch.normal(0, std_dev, size=param.grad.shape, device=param.grad.device))

# Dataset preparation
train_dataset = SequenceParatopeDataset(
    data_file=config['data_file_train'],
    sequence_file=config['sequence_file_train'],
    edge_file=config['edge_file_train'],
    max_len=config['max_len']
)

val_dataset = SequenceParatopeDataset(
    data_file=config['data_file_val'],
    sequence_file=config['sequence_file_val'],
    edge_file=config['edge_file_val'],
    max_len=config['max_len']
)

# Custom collate function
def custom_collate_fn(batch):
    sequences, pt, edge = zip(*batch)
    sequence_tensor = torch.stack(sequences)
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long)

    max_edges = max(edge_index.shape[0] for edge_index in edge)
    padded_edges = []
    for edge_index in edge:
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, :edge_index.shape[0]] = torch.tensor(edge_index.T, dtype=torch.long)
        padded_edges.append(edge_pad)
    padded_edges = torch.stack(padded_edges)

    return padded_edges, sequence_tensor, pt_tensor

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = CTMModel(
    vocab_size=config['vocab_size'],
    seq_len=config['max_len'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    dropout=config['dropout'],
    num_layers=config['num_layers'],
    num_gnn_layers=config['num_gnn_layers'],
    num_int_layers=config['num_int_layers'],
    num_classes=config['num_classes'],
    drop_path_rate=config['drop_path_rate']
)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs with DataParallel.")
    model = nn.DataParallel(model)
model = model.to(device)
# Load parameters
core_params = torch.load('isimodelNM_l0_g20_i8_do0.10_dpr0.10_lr0.0002_fold1_core.pth', map_location=device)
# Update model parameters
model_state = model.state_dict()
model_state.update(core_params)
model.load_state_dict(model_state)


#class_weight = torch.tensor([1.0, 1.0])
#class_weight = class_weight.to(device)
# Loss function, optimizer, scaler, scheduler
#criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight)
#criterion = FocalLossCE(gamma=2.0, weight=class_weight, reduction='mean', ignore_index=-1)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
scaler = GradScaler()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0
    current_gradient_noise_std = config['initial_gradient_noise_std'] * (0.9 ** epoch)

    for i, (padded_edges, sequence_tensor, contact_map_tensor) in enumerate(train_loader):
        # Zero the gradients on accumulation steps
        if i % config['accumulation_steps'] == 0:
            optimizer.zero_grad(set_to_none=True)

        # Move tensors to the correct device
        padded_edges = padded_edges.to(device)
        sequence_tensor = sequence_tensor.to(device)
        contact_map_tensor = contact_map_tensor.to(device)

        # Skip batch if ground truth contains NaN or Inf
        if torch.isnan(contact_map_tensor).any() or torch.isinf(contact_map_tensor).any():
            print("NaN or Inf found in target; skipping batch.")
            continue

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            # Forward pass: model returns logits and attention scores.
            # logits expected shape: [B, 2, seq_len, seq_len]
            logits, _ = model(
                sequences=sequence_tensor,
                padded_edges=padded_edges,
                return_attention=True,
                tied=False
            )

            # Reshape logits for CrossEntropyLoss:
            # Permute from [B, 2, seq_len, seq_len] to [B, seq_len, seq_len, 2]
            # then flatten to [N, 2] where N = B * seq_len * seq_len
            logits = logits.permute(0, 2, 3, 1).reshape(-1, 2)

            # Flatten the target contact map from [B, seq_len, seq_len] to [N]
            target = contact_map_tensor.view(-1)

            # Compute loss; note: no activation is applied because CrossEntropyLoss expects raw logits.
            loss = criterion(logits, target) / config['accumulation_steps']
        # Backward pass with mixed precision scaling
        scaler.scale(loss).backward()

        # Optionally add gradient noise
        if current_gradient_noise_std > 0:
            add_gradient_noise(model.parameters(), current_gradient_noise_std)

        # Gradient accumulation: update parameters every `accumulation_steps` iterations
        if (i + 1) % config['accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * config['accumulation_steps']


    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for padded_edges, sequence_tensor, contact_map_tensor in val_loader:
            padded_edges = padded_edges.to(device)
            sequence_tensor = sequence_tensor.to(device)
            contact_map_tensor = contact_map_tensor.to(device)
            if torch.isnan(contact_map_tensor).any() or torch.isinf(contact_map_tensor).any():
                print("NaN or Inf found in target; skipping batch.")
                continue
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                # Forward pass in validation (no attention output needed)
                logits, _ = model(sequences=sequence_tensor, padded_edges=padded_edges, return_attention=True, tied=False)
                # Reshape logits from [B, 2, seq_len, seq_len] to [N, 2]
                logits = logits.permute(0, 2, 3, 1).reshape(-1, 2)
                target = contact_map_tensor.view(-1)
                loss = criterion(logits, target)
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"isicmodelNMNew45_l{config['num_layers']}_g{config['num_gnn_layers']}_i{config['num_int_layers']}_dp{config['dropout']}.pth")
        print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}")

    if early_stop_counter >= config['early_stop_patience']:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

