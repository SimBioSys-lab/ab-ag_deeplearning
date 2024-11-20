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

base_config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_train_1200.npz',
    'pt_file': 'pt_train_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'num_classes': 2,
    'num_epochs': 50,  # Shortened for grid search
    'learning_rate': 0.003,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,
    'early_stop_patience': 10,
    'initial_gradient_noise_std': 0.05
}

# Parameter ranges for grid search
grid_params = {
    'num_layers': [1, 2, 4],
    'embed_dim': [128, 256, 1024],
    'num_heads': [4, 8, 16]
}

# Grid search results
results = []

# Load dataset and split into training and validation sets
print('Initializing dataset...', flush=True)
dataset = SequenceParatopeDataset(
    sequence_file=base_config['sequence_file'], pt_file=base_config['pt_file'], max_len=base_config['seq_len']
)
val_size = int(len(dataset) * base_config['validation_split'])
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=base_config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=base_config['batch_size'], shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Perform grid search
for num_layers in grid_params['num_layers']:
    for embed_dim in grid_params['embed_dim']:
        for num_heads in grid_params['num_heads']:
            print(f"Training with num_layers={num_layers}, embed_dim={embed_dim}, num_heads={num_heads}", flush=True)

            # Initialize model
            model = ParatopeModel(
                vocab_size=base_config['vocab_size'],
                seq_len=base_config['seq_len'],
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_classes=base_config['num_classes']
            )
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
                model = nn.DataParallel(model)
            model = model.to(device)

            # Initialize optimizer, scheduler, and criterion
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            optimizer = optim.Adam(model.parameters(), lr=base_config['learning_rate'])
            scaler = GradScaler()
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            # Training variables
            best_val_loss = float('inf')
            best_model_state = None
            early_stop_counter = 0

            # Training loop
            for epoch in range(base_config['num_epochs']):
                model.train()
                total_loss = 0

                # Decay gradient noise over epochs
                current_gradient_noise_std = base_config['initial_gradient_noise_std'] * (0.9 ** epoch)

                for seq, tgt in train_loader:
                    optimizer.zero_grad()
                    seq, tgt = seq.to(device), tgt.to(device)

                    # Skip invalid batches
                    if torch.isnan(seq).any() or torch.isinf(seq).any() or torch.isnan(tgt).any() or torch.isinf(tgt).any():
                        print("NaN or Inf found in input; skipping batch.")
                        continue

                    with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                        output_seq = model(seq)
                        output_seq = output_seq.view(-1, base_config['num_classes'])
                        tgt = tgt.view(-1)
                        loss = criterion(output_seq, tgt)

                    scaler.scale(loss).backward()

                    # Add gradient noise
                    if current_gradient_noise_std > 0:
                        for param in model.parameters():
                            if param.grad is not None:
                                noise = torch.normal(mean=0, std=current_gradient_noise_std, size=param.grad.shape, device=param.grad.device)
                                param.grad.add_(noise)

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), base_config['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                scheduler.step()
                print(f"Epoch [{epoch + 1}/{base_config['num_epochs']}], Training Loss: {avg_train_loss:.4f}", flush=True)

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for seq, tgt in val_loader:
                        seq, tgt = seq.to(device), tgt.to(device)

                        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                            output_seq = model(seq)
                            output_seq = output_seq.view(-1, base_config['num_classes'])
                            tgt = tgt.view(-1)
                            loss = criterion(output_seq, tgt)
                            val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Validation Loss: {avg_val_loss:.4f}", flush=True)

                # Track best model and apply early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict()
                    print(f"New best model with validation loss: {avg_val_loss:.4f}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{base_config['early_stop_patience']}")

                if early_stop_counter >= base_config['early_stop_patience']:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

            # Save results for this configuration
            results.append({
                'num_layers': num_layers,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'best_val_loss': best_val_loss
            })

            # Save the best model for this configuration
            if best_model_state is not None:
                model_path = f"model_layers_{num_layers}_dim_{embed_dim}_heads_{num_heads}.pth"
                torch.save(best_model_state, model_path)
                print(f"Best model saved to {model_path}", flush=True)

# Print grid search results
print("\nGrid Search Results:")
for res in results:
    print(res)

