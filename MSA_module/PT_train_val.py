import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import os
from Dataloader_all import SequenceParatopeDataset
from Models_new import ParatopeModel

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

config = {
    'batch_size': 4,
    'sequence_file': 'preprocessed_train_sequences.npz',
    'data_file': 'train_data.npz',
    'edge_file': 'train_edge_lists.npz',
    'max_len': 1200,
    'vocab_size': 22,
    'embed_dim': 256,
    'num_heads': 16,
    'num_layers': 4,
    'num_classes': 2,
    'num_epochs': 1000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,  # 10% for validation
    'early_stop_patience': 10,  # Stop if no improvement for 30 epochs
    'initial_gradient_noise_std': 0.05,  # Initial std for gradient noise
    'accumulation_steps': 2  # Gradient accumulation steps
}

print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# Helper function to add gradient noise
def add_gradient_noise(params, std_dev):
    """Add Gaussian noise to gradients."""
    for param in params:
        if param.grad is not None:
            noise = torch.normal(mean=0, std=std_dev, size=param.grad.shape, device=param.grad.device)
            param.grad.add_(noise)

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized edge lists and other tensors.
    """
    # Separate the batch into individual components
    sequences, ss, sasa, pt, sapt, edges = zip(*batch)

    # Stack fixed-size tensors
    sequence_tensor = torch.stack(sequences)
    ss_tensor = torch.stack(ss)
    sasa_tensor = torch.stack(sasa)
    pt_tensor = torch.stack(pt)
    sapt_tensor = torch.stack(sapt)

    # Keep edge lists as they are (list of tensors)
    edges_list = list(edges)

    return sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor, edges_list


# Dataset preparation
dataset = SequenceParatopeDataset(data_file=config['data_file'], sequence_file=config['sequence_file'], edge_file=config['edge_file'], max_len=config['max_len'])
val_size = int(len(dataset) * config['validation_split'])
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = ParatopeModel(
    vocab_size=config['vocab_size'],
    seq_len=config['max_len'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    num_classes=config['num_classes']
)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs with DataParallel.")
    model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
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

    # Decay the gradient noise standard deviation over epochs
    current_gradient_noise_std = config['initial_gradient_noise_std'] * (0.9 ** epoch)

    for i, (sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor, edges_tensor) in enumerate(train_loader):
        if i % config['accumulation_steps'] == 0:
            optimizer.zero_grad(set_to_none=True)

        sequence_tensor, pt_tensor = sequence_tensor.to(device), pt_tensor.to(device)

        # Skip batches with NaN or Inf
        if torch.isnan(sequence_tensor).any() or torch.isinf(sequence_tensor).any() or torch.isnan(pt_tensor).any() or torch.isinf(pt_tensor).any():
            print("NaN or Inf found in input; skipping batch.")
            continue

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output_seq = model(sequence_tensor)
            output_seq = output_seq.view(-1, config['num_classes'])
            pt_tensor = pt_tensor.view(-1)
            loss = criterion(output_seq, pt_tensor) / config['accumulation_steps']  # Normalize loss

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Add decayed gradient noise
        if current_gradient_noise_std > 0:
            add_gradient_noise(model.parameters(), current_gradient_noise_std)

        # Perform optimization step after accumulation_steps
        if (i + 1) % config['accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])  # Clip gradients
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * config['accumulation_steps']  # Un-normalize for logging

    avg_loss = total_loss / len(train_loader)
    scheduler.step()  # Adjust learning rate
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_loss:.4f}, Gradient Noise Std: {current_gradient_noise_std:.6f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor, edges_tensor  in val_loader:
            sequence_tensor, pt_tensor = sequence_tensor.to(device), pt_tensor.to(device)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output_seq = model(sequence_tensor)
                output_seq = output_seq.view(-1, config['num_classes'])
                pt_tensor = pt_tensor.view(-1)
                loss = criterion(output_seq, pt_tensor)
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Track the best model and apply early stopping
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

# Save the best model
if best_model_state is not None:
    torch.save(best_model_state, f'model_l{config['num_layers']}_d{config['embed_dim']}_h{config['num_heads']}_bs{config['batch_size']}x2.pth')
    print("Best model saved successfully.")
else:
    print("No best model found; check training configurations.")
