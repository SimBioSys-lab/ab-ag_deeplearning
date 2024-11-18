import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataloader_SS_SASA import SequenceSASADataset  # Assuming your dataset class is in 'Dataloader_SS_SASA.py'
from Models import SASAModel  # Assuming your SASAModel class is defined in 'Models'
from torch.amp import autocast, GradScaler
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration for the model and training
config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_1200.npz',
    'sasa_file': 'sasa_data.csv',
    'seq_len': 1200,   # Sequence length
    'vocab_size': 22,  # 20 amino acids + 1 unknown token + 1 padding token
    'embed_dim': 128,  # Embedding dimension
    'num_heads': 8,    # Number of attention heads
    'num_layers': 1,   # Number of layers in the model
    'num_epochs': 10,  # Number of training epochs
    'learning_rate': 0.01  # Learning rate for optimizer
}

# Load dataset and dataloader
print('Initializing dataset...', flush=True)
dataset = SequenceSASADataset(sequence_file=config['sequence_file'], sasa_file=config['sasa_file'], max_len=config['seq_len'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
print('Dataset loaded successfully.', flush=True)

# Initialize the model
model = SASAModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                  num_heads=config['num_heads'], num_layers=config['num_layers'])

# Use DataParallel to distribute across multiple GPUs, if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = nn.DataParallel(model)

# Send the model to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}", flush=True)

# Define loss function, optimizer, and gradient scaler
criterion = nn.MSELoss(reduction='none')  # Define MSE loss for continuous target values
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  # Define optimizer
scaler = GradScaler()  # Automatically detects if CUDA is available

# Training loop
for epoch in range(config['num_epochs']):
    print(f'Start epoch {epoch + 1}', flush=True)
    model.train()  # Set model to training mode
    total_loss = 0

    for seq, tgt in dataloader:
        optimizer.zero_grad()  # Reset gradients

        # Move data to the current device
        seq, tgt = seq.to(device), tgt.to(device)

        # Forward pass through the model
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output_seq = model(seq)
            
            # Create masks to ignore padding and replace padding with 0s
            mask = tgt < 0  # Assuming -1 is the padding value for the targets
            tgt_padded = torch.where(mask, tgt, torch.zeros_like(tgt)).float()

            # Compute loss while masking padding positions
            loss = (criterion(output_seq, tgt_padded) * mask.float()).mean()
            total_loss += loss.item()

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}', flush=True)

print("Training finished!", flush=True)

# After training is done, export the learned layers
pretrained_layers = {
    'SelfAttention': model.module.self_attention.state_dict() if hasattr(model, 'module') else model.self_attention.state_dict(),
}

# Save the layers to a file
torch.save(pretrained_layers, 'pretrained_SelfAttention_SASA.pth')
print("Model layers saved successfully.", flush=True)

