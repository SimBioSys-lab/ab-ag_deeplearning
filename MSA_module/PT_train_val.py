import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Dataloader_SS_SASA import SequenceParatopeDataset
from Models import ParatopeModel
from torch.amp import autocast, GradScaler
import os

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_train_1200.npz',
    'pt_file': 'pt_train_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 2,
    'num_classes': 2,
    'num_epochs': 1000,
    'learning_rate': 0.003,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,  # 10% for validation
    'early_stop_patience': 30  # Stop if no improvement for 20 epochs
}
print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# Load dataset and split into training and validation sets
dataset = SequenceParatopeDataset(sequence_file=config['sequence_file'], pt_file=config['pt_file'], max_len=config['seq_len'])
val_size = int(len(dataset) * config['validation_split'])
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, criterion, optimizer, scaler, and scheduler
model = ParatopeModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                      num_heads=config['num_heads'], num_layers=config['num_layers'], num_classes=config['num_classes'])
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
    for seq, tgt in train_loader:
        optimizer.zero_grad()
        seq, tgt = seq.to(device), tgt.to(device)
        if torch.isnan(seq).any() or torch.isinf(seq).any() or torch.isnan(tgt).any() or torch.isinf(tgt).any():
            print("NaN or Inf found in input; skipping batch.")
            continue
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output_seq = model(seq)
            output_seq = output_seq.view(-1, config['num_classes'])
            tgt = tgt.view(-1)
            loss = criterion(output_seq, tgt)
            total_loss += loss.item()

        # Backward pass, gradient clipping, and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale gradients for clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()

    avg_loss = total_loss / len(train_loader)
    scheduler.step()  # Adjust learning rate
    print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Training Loss: {avg_loss:.4f}', flush=True)

    # Validation
    model.eval()
    val_loss = 0
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
    print(f'Validation Loss: {avg_val_loss:.4f}', flush=True)

    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    print(f"Peak Memory Usage: {peak_memory:.2f} GB")

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        print(f'New best model found with validation loss: {avg_val_loss:.4f}', flush=True)
        early_stop_counter = 0  # Reset counter if validation loss improves
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}", flush=True)

    # Stop training if validation loss has not improved for 'early_stop_patience' epochs
    if early_stop_counter >= config['early_stop_patience']:
        print(f"Early stopping triggered after {epoch + 1} epochs.", flush=True)
        break

# Save the best model state
if best_model_state is not None:
    torch.save(best_model_state, 'best_paratope_model_2l.pth')
    print("Best model saved successfully.", flush=True)
else:
    print("No best model found; check training configurations.", flush=True)

# Access individual layer parameters
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")

