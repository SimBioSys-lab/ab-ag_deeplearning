import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import os
import numpy as np
from Dataloader_s import SequenceParatopeDataset
from Models_new import ClassificationSModel

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

config = {
    'batch_size': 4,
    'absequence_file': 'antibody_train_sequences_1300.npz',
    'abdata_file': 'antibody_train_interfaces_1300.npz',
    'abedge_file': 'antibody_train_edges_1300.npz',
    'abmax_len': 1300,
    'agsequence_file': 'antigen_train_sequences_2500.npz',
    'agdata_file': 'antigen_train_interfaces_2500.npz',
    'agedge_file': 'antigen_train_edges_2500.npz',
    'agmax_len': 2500,
    'vocab_size': 23,
    'embed_dim': 256,
    'num_heads': 16,
    'dropout': 0.0,
    'num_layers': 1,
    'num_gnn_layers': 1,
    'num_int_layers': 1,
    'num_classes': 2,
    'num_epochs': 1000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.1,
    'early_stop_patience': 10,
    'initial_gradient_noise_std': 0.05,
    'accumulation_steps': 2,
    'train_split': 0.8
}

# Helper function to add gradient noise
def add_gradient_noise(params, std_dev):
    for param in params:
        if param.grad is not None:
            noise = torch.normal(mean=0, std=std_dev, size=param.grad.shape, device=param.grad.device)
            param.grad.add_(noise)

# Dataset preparation
dataset = SequenceParatopeDataset(
    abdata_file=config['abdata_file'],
    absequence_file=config['absequence_file'],
    abedge_file=config['abedge_file'],
    abmax_len=config['abmax_len'],
    agdata_file=config['agdata_file'],
    agsequence_file=config['agsequence_file'],
    agedge_file=config['agedge_file'],
    agmax_len=config['agmax_len']
)

# Split the dataset into train and test sets
train_size = int(len(dataset) * config['train_split'])
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Custom collate function for batching data
def custom_collate_fn(batch):
    absequences, abpt, abedge, agsequences, agpt, agedge = zip(*batch)
    absequence_tensor = torch.stack(absequences)
    abpt_tensor = torch.tensor(np.array(abpt), dtype=torch.long)
    agsequence_tensor = torch.stack(agsequences)
    agpt_tensor = torch.tensor(np.array(agpt), dtype=torch.long)

    def pad_edges(edges):
        max_edges = max(edge.shape[1] for edge in edges)
        padded_edges = [
            torch.cat([edge, -torch.ones((2, max_edges - edge.shape[1]), dtype=torch.long)], dim=1)
            for edge in edges
        ]
        return torch.stack(padded_edges)

    abpadded_edges = pad_edges(abedge)
    agpadded_edges = pad_edges(agedge)

    return abpadded_edges, absequence_tensor, abpt_tensor, agpadded_edges, agsequence_tensor, agpt_tensor

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False)

# Initialize the model
model = ClassificationSModel(
    vocab_size=config['vocab_size'],
    seq_len=config['abmax_len'] + config['agmax_len'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    dropout=config['dropout'],
    num_layers=config['num_layers'],
    num_gnn_layers=config['num_gnn_layers'],
    num_int_layers=config['num_int_layers'],
    num_classes=config['num_classes']
).to(device)

# Loss, optimizer, and gradient scaling setup
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
scaler = GradScaler()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop with train-test split
best_val_loss = float('inf')
best_model_state = None
early_stop_counter = 0

for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0
    current_gradient_noise_std = config['initial_gradient_noise_std'] * (0.9 ** epoch)

    for i, (abpadded_edges, absequence_tensor, abpt_tensor, agpadded_edges, agsequence_tensor, agpt_tensor) in enumerate(train_loader):
        if i % config['accumulation_steps'] == 0:
            optimizer.zero_grad(set_to_none=True)

        absequence_tensor, abpt_tensor = absequence_tensor.to(device), abpt_tensor.to(device)
        abpadded_edges = abpadded_edges.to(device)
        agsequence_tensor, agpt_tensor = agsequence_tensor.to(device), agpt_tensor.to(device)
        agpadded_edges = agpadded_edges.to(device)

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output_seq = model(absequence_tensor, agsequence_tensor, abpadded_edges, agpadded_edges)
            output_seq = output_seq.view(-1, config['num_classes'])
            pt_tensor = torch.cat([abpt_tensor.view(-1), agpt_tensor.view(-1)])
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

    # Test loop for validation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for abpadded_edges, absequence_tensor, abpt_tensor, agpadded_edges, agsequence_tensor, agpt_tensor in test_loader:
            absequence_tensor, abpt_tensor = absequence_tensor.to(device), abpt_tensor.to(device)
            abpadded_edges = abpadded_edges.to(device)
            agsequence_tensor, agpt_tensor = agsequence_tensor.to(device), agpt_tensor.to(device)
            agpadded_edges = agpadded_edges.to(device)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output_seq = model(absequence_tensor, agsequence_tensor, abpadded_edges, agpadded_edges)
                output_seq = output_seq.view(-1, config['num_classes'])
                pt_tensor = torch.cat([abpt_tensor.view(-1), agpt_tensor.view(-1)])
                loss = criterion(output_seq, pt_tensor)
                test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Validation Loss: {avg_test_loss:.4f}")

    # Early stopping and model checkpointing
    if avg_test_loss < best_val_loss:
        best_val_loss = avg_test_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print(f"New best model with validation loss: {avg_test_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stop counter: {early_stop_counter}/{config['early_stop_patience']}")

    if early_stop_counter >= config['early_stop_patience']:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

# Save the best model
if best_model_state is not None:
    torch.save(best_model_state, f'mcsmodel_final.pth')
    print("Best model saved successfully.")
else:
    print("No best model saved; check the training configurations.")

