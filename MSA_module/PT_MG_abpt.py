import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler
import os
import numpy as np
from sklearn.model_selection import KFold
from Dataloader_itf import SequenceParatopeDataset
from Models_new import ClassificationModel

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()  # Optional: Reset memory tracking

config = {
    'batch_size': 4,
    'sequence_file': 'padded_sequences_train_3000.npz',
    'data_file': 'antibody_train_interfaces_aligned_3000.npz',
    'edge_file': 'padded_edges_train_3000.npz',
    'max_len': 3000,
    'vocab_size': 23,
    'embed_dim': 256,
    'num_heads': 16,
    'dropout': 0.0,
    'num_layers': 1,
    'num_gnn_layers': 2,
    'num_int_layers': 1,
    'num_classes': 2,
    'num_epochs': 1000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,
    'early_stop_patience': 10,
    'initial_gradient_noise_std': 0.05,
    'accumulation_steps': 2
}

print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()  # Optional: Reset memory tracking
# Helper function to add gradient noise
def add_gradient_noise(params, std_dev):
    """Add Gaussian noise to gradients."""
    for param in params:
        if param.grad is not None:
            noise = torch.normal(mean=0, std=std_dev, size=param.grad.shape, device=param.grad.device)
            param.grad.add_(noise)

# Dataset preparation
dataset = SequenceParatopeDataset(
    data_file=config['data_file'],
    sequence_file=config['sequence_file'],
    edge_file=config['edge_file'],
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

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
dataset_indices = np.arange(len(dataset))
fold_results = []

for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_indices), 1):
    print(f"\nStarting Fold {fold}/5...")
    
    # Create train and validation datasets for the current fold
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False
    )

    # Model initialization
    model = ClassificationModel(
        vocab_size=config['vocab_size'],
        seq_len=config['max_len'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
        num_gnn_layers=config['num_gnn_layers'],
        num_int_layers=config['num_int_layers'],
        num_classes=config['num_classes']
    )
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss function, optimizer, scaler, scheduler
#    class_weights_tensor = torch.tensor([0.6, 3.5], dtype=torch.float).to(device)
#    criterion = nn.CrossEntropyLoss(ignore_index=-1,weight=class_weights_tensor)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Track the best model performance and early stopping criteria
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        current_gradient_noise_std = config['initial_gradient_noise_std'] * (0.9 ** epoch)

        for i, (padded_edges, sequence_tensor, pt_tensor) in enumerate(train_loader):
            if i % config['accumulation_steps'] == 0:
                optimizer.zero_grad(set_to_none=True)

            sequence_tensor, pt_tensor = sequence_tensor.to(device), pt_tensor.to(device)
            padded_edges = padded_edges.to(device)

            if torch.isnan(pt_tensor).any() or torch.isinf(pt_tensor).any():
                print("NaN or Inf found in input; skipping batch.")
                continue

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output_seq = model(sequence_tensor, padded_edges)
                output_seq = output_seq.view(-1, config['num_classes'])
                pt_tensor = pt_tensor.view(-1)
                loss = criterion(output_seq, pt_tensor) / config['accumulation_steps']

            scaler.scale(loss).backward()

            if current_gradient_noise_std > 0:
                add_gradient_noise(model.parameters(), current_gradient_noise_std)

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
            for padded_edges, sequence_tensor, pt_tensor in val_loader:
                padded_edges, sequence_tensor, pt_tensor = padded_edges.to(device), sequence_tensor.to(device), pt_tensor.to(device)
                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    output_seq = model(sequence_tensor, padded_edges)
                    output_seq = output_seq.view(-1, config['num_classes'])
                    pt_tensor = pt_tensor.view(-1)
                    loss = criterion(output_seq, pt_tensor)
                    val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"New best model found with validation loss: {avg_val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}")

        if early_stop_counter >= config['early_stop_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Save the best model for the current fold
    if best_model_state is not None:
        torch.save(best_model_state, f'mcabptmodel_fold{fold}_l{config["num_layers"]}_g{config["num_gnn_layers"]}_i{config["num_int_layers"]}_dp{config["dropout"]}.pth')
        print(f"Best model for fold {fold} saved successfully.")
    else:
        print(f"No best model found for fold {fold}; check training configurations.")

    # Store the validation loss for this fold
    fold_results.append(best_val_loss)

# Print overall cross-validation results
print("\nCross-validation results:")
for fold, val_loss in enumerate(fold_results, 1):
    print(f"Fold {fold}: Validation Loss = {val_loss:.4f}")
print(f"Average Validation Loss: {np.mean(fold_results):.4f}")

