import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from Dataloader_SS_SASA import SequenceParatopeDataset, SequenceSASADataset, SequenceSecondaryStructureDataset
from Models import UnifiedModel
from torch.amp import autocast, GradScaler
import os

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_train_1200.npz',
    'pt_file': 'pt_train_data.csv',
    'sasa_file': 'sasa_train_data.csv',
    'ss_file': 'ss_train_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 3,
    'num_classes_pt': 2,
    'num_classes_ss': 8,
    'num_epochs': 200,
    'learning_rate': 0.003,
    'n_splits': 5,
    'max_grad_norm': 0.5,
    'early_stop_patience': 20  # Early stopping patience in epochs
}

print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# Load datasets for each task
paratope_dataset = SequenceParatopeDataset(sequence_file=config['sequence_file'], pt_file=config['pt_file'], max_len=config['seq_len'])
sasa_dataset = SequenceSASADataset(sequence_file=config['sequence_file'], sasa_file=config['sasa_file'], max_len=config['seq_len'])
ss_dataset = SequenceSecondaryStructureDataset(sequence_file=config['sequence_file'], ss_file=config['ss_file'], max_len=config['seq_len'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cross-validation loop
kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
fold_results = []

# Initialize the unified model
model = UnifiedModel(
    vocab_size=config['vocab_size'],
    seq_len=config['seq_len'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    num_classes_ss=config['num_classes_ss'],
    num_classes_pt=config['num_classes_pt']
)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs with DataParallel.")
    model = nn.DataParallel(model)
model = model.to(device)

# Initialize loss functions, optimizer, scaler, and scheduler
criterion_pt = nn.CrossEntropyLoss(ignore_index=-1)
criterion_ss = nn.CrossEntropyLoss(ignore_index=-1)
criterion_sasa = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
scaler = GradScaler()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

# Track the best model performance and early stopping criteria
best_val_loss = float('inf')
best_model_state = None
early_stop_counter = 0

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(paratope_dataset)):
    print(f'Starting fold {fold + 1}/{config["n_splits"]}', flush=True)
    torch.cuda.empty_cache()  # Clear cache between folds

    # Create data loaders for training and validation
    train_paratope = DataLoader(Subset(paratope_dataset, train_idx), batch_size=config['batch_size'], shuffle=True)
    val_paratope = DataLoader(Subset(paratope_dataset, val_idx), batch_size=config['batch_size'], shuffle=False)
    train_sasa = DataLoader(Subset(sasa_dataset, train_idx), batch_size=config['batch_size'], shuffle=True)
    val_sasa = DataLoader(Subset(sasa_dataset, val_idx), batch_size=config['batch_size'], shuffle=False)
    train_ss = DataLoader(Subset(ss_dataset, train_idx), batch_size=config['batch_size'], shuffle=True)
    val_ss = DataLoader(Subset(ss_dataset, val_idx), batch_size=config['batch_size'], shuffle=False)

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        pt_loss = 0
        for (seq_pt, tgt_pt), (seq_sasa, tgt_sasa), (seq_ss, tgt_ss) in zip(train_paratope, train_sasa, train_ss):
            optimizer.zero_grad()
            seq_pt, tgt_pt = seq_pt.to(device), tgt_pt.to(device)
            seq_sasa, tgt_sasa = seq_sasa.to(device), tgt_sasa.to(device)
            seq_ss, tgt_ss = seq_ss.to(device), tgt_ss.to(device)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                # Forward pass for each task
                sasa_output, ss_output, pt_output = model(seq_pt)

                # Loss calculations for each task
                loss_sasa = (criterion_sasa(sasa_output, tgt_sasa.float()) * (tgt_sasa >= 0).float()).mean()
                loss_ss = criterion_ss(ss_output.view(-1, config['num_classes_ss']), tgt_ss.view(-1))
                loss_pt = criterion_pt(pt_output.view(-1, config['num_classes_pt']), tgt_pt.view(-1))

                # Combine losses
                loss = loss_sasa * 8 + loss_ss * 0.5 + loss_pt
                total_loss += loss.item()
                pt_loss += loss_pt.item()
            # Backward pass, gradient clipping, and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()

        avg_loss = total_loss / len(train_paratope)
        avg_pt_loss = pt_loss / len(train_paratope)
        scheduler.step()
        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{config["num_epochs"]}], Training Loss: {avg_loss:.4f}, PT Loss: {avg_pt_loss:.4f}', flush=True)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (seq_pt, tgt_pt), (seq_sasa, tgt_sasa), (seq_ss, tgt_ss) in zip(val_paratope, val_sasa, val_ss):
                seq_pt, tgt_pt = seq_pt.to(device), tgt_pt.to(device)
                seq_sasa, tgt_sasa = seq_sasa.to(device), tgt_sasa.to(device)
                seq_ss, tgt_ss = seq_ss.to(device), tgt_ss.to(device)

                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    sasa_output, ss_output, pt_output = model(seq_pt)
                    loss_sasa = (criterion_sasa(sasa_output, tgt_sasa.float()) * (tgt_sasa >= 0).float()).mean()
                    loss_ss = criterion_ss(ss_output.view(-1, config['num_classes_ss']), tgt_ss.view(-1))
                    loss_pt = criterion_pt(pt_output.view(-1, config['num_classes_pt']), tgt_pt.view(-1))

                    val_loss += (loss_sasa + loss_ss + loss_pt).item()

        avg_val_loss = val_loss / len(val_paratope)
        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{config["num_epochs"]}], Validation Loss: {avg_val_loss:.4f}', flush=True)

        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f'New best model found at fold {fold + 1}, epoch {epoch + 1} with validation loss: {avg_val_loss:.4f}', flush=True)
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}", flush=True)

        # Stop training if validation loss hasn't improved for `early_stop_patience` epochs
        if early_stop_counter >= config['early_stop_patience']:
            print(f"Early stopping triggered after epoch {epoch + 1} in fold {fold + 1}.", flush=True)
            break

    fold_results.append(best_val_loss)

# Report average validation loss across folds
average_loss = sum(fold_results) / config['n_splits']
print(f'Average Validation Loss across {config["n_splits"]} folds: {average_loss:.4f}')

# Save the best model state
if best_model_state is not None:
    torch.save(best_model_state, 'best_unified_model.pth')
    print("Best model saved successfully.", flush=True)

