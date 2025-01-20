import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import os
from Dataloader_all import SequenceParatopeDataset
from Models_new import CoreModelWithGNN_MT
import numpy as np

# Configuration for model and training
torch.backends.cudnn.benchmark = True

config = {
    'batch_size': 4,
    'sequence_file': 'preprocessed_train_sequences.npz',
    'data_file': 'train_data.npz',
    'edge_file': 'train_edge_lists.npz',
    'max_len': 1200,
    'vocab_size': 22,
    'embed_dim': 256,
    'num_heads': 16,
    'num_layers': 1,
    'num_gnn_layers': 3,
    'pt_num_classes': 2,  # For Paratope classification
    'ss_num_classes': 8,  # For Secondary Structure classification
    'num_epochs': 1000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,  # 10% for validation
    'early_stop_patience': 10,
    'accumulation_steps': 2
}

print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# Dataset preparation
dataset = SequenceParatopeDataset(
    data_file=config['data_file'],
    sequence_file=config['sequence_file'],
    edge_file=config['edge_file'],
    max_len=config['max_len']
)
val_size = int(len(dataset) * config['validation_split'])
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def custom_collate_fn(batch):
    """
    Custom collate function to pad edge indices dynamically based on the longest edge index in the batch.
    """
    sequences, ss, sasa, pt, sapt, edge = zip(*batch)

    # Stack sequences into a single tensor
    sequence_tensor = torch.stack(sequences)

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
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, :edge_index.shape[0]] = torch.tensor(edge_index.T, dtype=torch.long)
        padded_edges.append(edge_pad)
    padded_edges = torch.stack(padded_edges)

    return padded_edges, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = CoreModelWithGNN_MT(
    vocab_size=config['vocab_size'],
    seq_len=config['max_len'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    num_gnn_layers=config['num_gnn_layers'],
    pt_num_classes=config['pt_num_classes'],
    ss_num_classes=config['ss_num_classes']
)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs with DataParallel.")
    model = nn.DataParallel(model)
model = model.to(device)

# Loss functions and optimizer
criterion_pt = nn.CrossEntropyLoss(ignore_index=-1)
criterion_ss = nn.CrossEntropyLoss(ignore_index=-1)
criterion_sasa = nn.MSELoss(reduction='none')  # For per-element masking
criterion_sapt = nn.MSELoss()

# Learnable weights for multitask loss
task_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0], device=device), requires_grad=True)
optimizer = optim.Adam([{'params': model.parameters()}, {'params': task_weights}], lr=config['learning_rate'])
scaler = GradScaler()

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
best_val_loss = float('inf')
best_model_state = None
early_stop_counter = 0

for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0

    for i, (padded_edges, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        # Move data to device
        padded_edges, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor = (
            padded_edges.to(device), sequence_tensor.to(device), ss_tensor.to(device),
            sasa_tensor.to(device), pt_tensor.to(device), sapt_tensor.to(device)
        )

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            # Forward pass
            pt_pred, ss_pred, sasa_pred, sapt_pred = model(sequence_tensor, padded_edges)

            # Calculate individual losses
            loss_pt = criterion_pt(pt_pred.view(-1, config['pt_num_classes']), pt_tensor.view(-1))
            loss_ss = criterion_ss(ss_pred.view(-1, config['ss_num_classes']), ss_tensor.view(-1))

            # Mask padding positions for SASA loss
            sasa_mask = sasa_tensor != -1
            sasa_target_padded = torch.where(sasa_mask, sasa_tensor, torch.zeros_like(sasa_tensor))
            loss_sasa = (criterion_sasa(sasa_pred, sasa_target_padded) * sasa_mask.float()).sum() / sasa_mask.sum().clamp(min=1)

            # SAPT loss
            loss_sapt = criterion_sapt(sapt_pred, sapt_tensor)

            # Total loss with learnable weights
            task_loss = torch.stack([loss_pt, loss_ss, loss_sasa, loss_sapt])
            total_task_loss = torch.dot(task_weights, task_loss)

        # Backward pass with gradient scaling
        scaler.scale(total_task_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += total_task_loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_loss:.4f}, Task Weights: {task_weights.detach().cpu().numpy()}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for padded_edges, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor in val_loader:
            padded_edges, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor = (
                padded_edges.to(device), sequence_tensor.to(device), ss_tensor.to(device),
                sasa_tensor.to(device), pt_tensor.to(device), sapt_tensor.to(device)
            )

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                pt_pred, ss_pred, sasa_pred, sapt_pred = model(sequence_tensor, padded_edges)

                loss_pt = criterion_pt(pt_pred.view(-1, config['pt_num_classes']), pt_tensor.view(-1))
                loss_ss = criterion_ss(ss_pred.view(-1, config['ss_num_classes']), ss_tensor.view(-1))

                sasa_mask = sasa_tensor != -1
                sasa_target_padded = torch.where(sasa_mask, sasa_tensor, torch.zeros_like(sasa_tensor))
                loss_sasa = (criterion_sasa(sasa_pred, sasa_target_padded) * sasa_mask.float()).sum() / sasa_mask.sum().clamp(min=1)

                loss_sapt = criterion_sapt(sapt_pred, sapt_tensor)
                task_loss = torch.stack([loss_pt, loss_ss, loss_sasa, loss_sapt])
                total_task_loss = torch.dot(task_weights, task_loss)

                val_loss += total_task_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print(f"New best model found with validation loss: {avg_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}")

    if early_stop_counter >= config['early_stop_patience']:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

if best_model_state is not None:
    torch.save(best_model_state, 'best_multitask_model.pth')
    print("Best model saved.")
else:
    print("No best model found.")

