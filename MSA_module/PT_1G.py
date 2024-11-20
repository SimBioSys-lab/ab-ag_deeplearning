import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Dataloader_SS_SASA import SequenceParatopeDataset
from Models import ParatopeModel
from torch.amp import autocast, GradScaler
import os
import random

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

config = {
    'batch_size': 6,
    'sequence_file': 'preprocessed_seq_ab_train_1200.npz',
    'pt_file': 'pt_train_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 2,
    'num_classes': 2,
    'num_epochs': 1000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,
    'early_stop_patience': 30,
    'gradient_accumulation_steps': 4,
    'gradient_noise_std': 0.01,  # Standard deviation for gradient noise
    'checkpoint_path': 'checkpoint.pt'  # Path for saving checkpoints
}

print(config)
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Helper function to add gradient noise
def add_gradient_noise(params, std_dev):
    """Add Gaussian noise to gradients."""
    for param in params:
        if param.grad is not None:
            noise = torch.normal(mean=0, std=std_dev, size=param.grad.shape, device=param.grad.device)
            param.grad.add_(noise)

# Cleanup function
def clean_up():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Dataset preparation
dataset = SequenceParatopeDataset(
    sequence_file=config['sequence_file'], pt_file=config['pt_file'], max_len=config['seq_len']
)
val_size = int(len(dataset) * config['validation_split'])
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = ParatopeModel(
    vocab_size=config['vocab_size'],
    seq_len=config['seq_len'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    num_classes=config['num_classes']
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
scaler = GradScaler()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Track the best model performance and early stopping criteria
best_val_loss = float('inf')
best_model_state = None
early_stop_counter = 0
last_epoch = 0

# Load checkpoint if available
if os.path.exists(config['checkpoint_path']):
    print("Loading checkpoint...")
    checkpoint = torch.load(config['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    best_val_loss = checkpoint['best_val_loss']
    last_epoch = checkpoint['epoch']
    print(f"Resuming from epoch {last_epoch + 1} with best validation loss: {best_val_loss:.4f}")

# Training loop
clean_up()  # Pre-training cleanup
for epoch in range(last_epoch + 1, config['num_epochs']):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()  # Reset gradients

    for step, (seq, tgt) in enumerate(train_loader):
        seq, tgt = seq.to(device), tgt.to(device)

        # Skip batches with NaN or Inf
        if torch.isnan(seq).any() or torch.isinf(seq).any() or torch.isnan(tgt).any() or torch.isinf(tgt).any():
            print("NaN or Inf found in input; skipping batch.")
            continue

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output_seq = model(seq)
            output_seq = output_seq.view(-1, config['num_classes'])
            tgt = tgt.view(-1)
            loss = criterion(output_seq, tgt) / config['gradient_accumulation_steps']

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Add gradient noise
        if config['gradient_noise_std'] > 0:
            add_gradient_noise(model.parameters(), config['gradient_noise_std'])

        # Gradient accumulation
        if (step + 1) % config['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])  # Clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config['gradient_accumulation_steps']

    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq, tgt in val_loader:
            seq, tgt = seq.to(device), tgt.to(device)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output_seq = model(seq)
                output_seq = output_seq.view(-1, config['num_classes'])
                tgt = tgt.view(-1)
                loss = criterion(output_seq, tgt)
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Track the best model and apply early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        print(f"New best model found with Validation Loss: {avg_val_loss:.4f}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}")

    if early_stop_counter >= config['early_stop_patience']:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, config['checkpoint_path'])
    print(f"Checkpoint saved at epoch {epoch + 1}.")

# Save the best model
if best_model_state is not None:
    torch.save(best_model_state, 'best_paratope_model_with_noise_2l.pth')
    print("Best model saved successfully.")
else:
    print("No best model found; check training configurations.")

clean_up()  # Post-training cleanup

