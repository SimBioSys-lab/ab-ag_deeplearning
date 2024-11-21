import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Dataloader_SS_SASA import SequenceParatopeDataset, SequenceSAPTDataset
from Models import PTSAPTModel
from torch.amp import autocast, GradScaler
import os

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_train_1200.npz',
    'pt_file': 'pt_train_data.csv',
    'sapt_file': 'sapt_train_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 1,
    'num_classes_pt': 2,
    'num_epochs': 1000,
    'learning_rate': 0.003,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,  # 10% for validation
    'early_stop_patience': 30,  # Stop if no improvement for 30 epochs
    'initial_gradient_noise_std': 0.05  # Initial standard deviation for gradient noise
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

# Dataset preparation
paratope_dataset = SequenceParatopeDataset(sequence_file=config['sequence_file'], pt_file=config['pt_file'], max_len=config['seq_len'])
sapt_dataset = SequenceSAPTDataset(sequence_file=config['sequence_file'], sapt_file=config['sapt_file'], max_len=config['seq_len'])

val_size = int(len(paratope_dataset) * config['validation_split'])
train_size = len(paratope_dataset) - val_size

train_paratope, val_paratope = random_split(paratope_dataset, [train_size, val_size])
train_sapt, val_sapt = random_split(sapt_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(list(zip(train_paratope, train_sapt)), batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(list(zip(val_paratope, val_sapt)), batch_size=config['batch_size'], shuffle=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = PTSAPTModel(
    vocab_size=config['vocab_size'],
    seq_len=config['seq_len'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    num_classes_pt=config['num_classes_pt']
)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs with DataParallel.")
    model = nn.DataParallel(model)
model = model.to(device)

criterion_pt = nn.CrossEntropyLoss(ignore_index=-1)  # For paratope classification
criterion_sapt = nn.MSELoss()  # For SAPT regression
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

    for (seq_pt, tgt_pt), (seq_sapt, tgt_sapt) in train_loader:
        optimizer.zero_grad()
        seq_pt, tgt_pt = seq_pt.to(device), tgt_pt.to(device)
        seq_sapt, tgt_sapt = seq_sapt.to(device), tgt_sapt.to(device)

        # Skip invalid batches
        if torch.isnan(seq_pt).any() or torch.isinf(seq_pt).any() or torch.isnan(tgt_pt).any() or torch.isinf(tgt_pt).any():
            print("NaN or Inf found in input; skipping batch.")
            continue

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            sapt_output, pt_output = model(seq_pt)
            
            # Compute losses
            loss_sapt = criterion_sapt(sapt_output, tgt_sapt)
            loss_pt = criterion_pt(pt_output.view(-1, config['num_classes_pt']), tgt_pt.view(-1))

            # Combine losses
            loss = loss_sapt + loss_pt

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Add gradient noise
        if current_gradient_noise_std > 0:
            add_gradient_noise(model.parameters(), current_gradient_noise_std)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])  # Clip gradients
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step()  # Adjust learning rate
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_loss:.4f}, Gradient Noise Std: {current_gradient_noise_std:.6f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (seq_pt, tgt_pt), (seq_sapt, tgt_sapt) in val_loader:
            seq_pt, tgt_pt = seq_pt.to(device), tgt_pt.to(device)
            seq_sapt, tgt_sapt = seq_sapt.to(device), tgt_sapt.to(device)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                sapt_output, pt_output = model(seq_pt)

                # Compute losses
                loss_sapt = criterion_sapt(sapt_output, tgt_sapt)
                loss_pt = criterion_pt(pt_output.view(-1, config['num_classes_pt']), tgt_pt.view(-1))

                val_loss += (loss_sapt + loss_pt).item()

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
    torch.save(best_model_state, 'best_ptsapt_model.pth')
    print("Best model saved successfully.")
else:
    print("No best model found; check training configurations.")

