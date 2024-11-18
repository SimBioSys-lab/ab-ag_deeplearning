import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Dataloader_SS_SASA import SequenceSASADataset
from Models import SASAModel
from torch.amp import autocast, GradScaler
import os

# Set environment and model configuration
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_train_1200.npz',
    'sasa_file': 'sasa_train_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 1,
    'num_epochs': 1000,
    'learning_rate': 0.003,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,      # 10% for validation
    'early_stop_patience': 30     # Early stopping patience (epochs)
}

# Load dataset and split into training and validation sets
print('Initializing dataset...', flush=True)
dataset = SequenceSASADataset(sequence_file=config['sequence_file'], sasa_file=config['sasa_file'], max_len=config['seq_len'])
val_size = int(len(dataset) * config['validation_split'])
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, optimizer, scaler, and scheduler
model = SASAModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                  num_heads=config['num_heads'], num_layers=config['num_layers'])
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss(reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
scaler = GradScaler()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# Track the best model performance and early stopping criteria
best_val_loss = float('inf')
best_model_state = None
early_stop_counter = 0

# Training loop with early stopping
for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0
    for seq, tgt in train_loader:
        optimizer.zero_grad()
        seq, tgt = seq.to(device), tgt.to(device)
        if torch.isnan(seq).any() or torch.isinf(seq).any() or torch.isnan(tgt).any() or torch.isinf(tgt).any():
            print("NaN or Inf found in input; skipping batch.")
            continue
        # Forward pass with mixed precision
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output_seq = model(seq)

            # Mask padding positions and compute masked loss
            mask = tgt >= 0  # Assuming -1 is the padding value for targets
            tgt_padded = torch.where(mask, tgt, torch.zeros_like(tgt)).float()
            loss = (criterion(output_seq, tgt_padded) * mask.float()).sum() / mask.sum().clamp(min=1)  # Normalized loss

            # Check for NaN
            if torch.isnan(loss):
                print("NaN detected in loss; skipping this batch.")
                continue
            total_loss += loss.item()

        # Backward pass and optimization with gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()

    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Training Loss: {avg_loss:.4f}', flush=True)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seq, tgt in val_loader:
            seq, tgt = seq.to(device), tgt.to(device)
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output_seq = model(seq)
                
                # Mask padding positions in validation loss
                mask = tgt != -1
                tgt_padded = torch.where(mask, tgt, torch.zeros_like(tgt)).float()
                loss = (criterion(output_seq, tgt_padded) * mask.float()).sum() / mask.sum().clamp(min=1)  # Normalized loss
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}', flush=True)

    # Early stopping and saving the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        print(f'New best model found with validation loss: {avg_val_loss:.4f}', flush=True)
        early_stop_counter = 0  # Reset early stopping counter if validation loss improves
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}", flush=True)

    # Stop training if validation loss hasn't improved for 'early_stop_patience' epochs
    if early_stop_counter >= config['early_stop_patience']:
        print(f"Early stopping triggered after {epoch + 1} epochs.", flush=True)
        break

# Save the best model state
if best_model_state is not None:
    torch.save(best_model_state, 'best_sasa_model.pth')
    print("Best model saved successfully.", flush=True)
else:
    print("No best model found; check training configurations.", flush=True)

