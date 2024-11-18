import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader_att import PreprocessedMSADataset  # Assuming your dataset class is in 'DataLoader_att.py'
from Models import MSAModel  # Assuming your MSAModel class is defined in 'Models'
from torch.amp import autocast, GradScaler

# Hyperparameters and file paths
config = {
    'batch_size': 8,
    'csv_file': 'preprocessed_data_1200.csv',
    'seq_len': 1200,   # Sequence length
    'vocab_size': 22,  # 20 amino acids + 1 unknown token + 1 padding token
    'embed_dim': 128,  # Embedding dimension
    'num_heads': 8,    # Number of attention heads
    'num_layers': 1,
    'num_epochs': 10,
    'learning_rate': 0.01
}

# Load dataset and dataloader
print('0', flush=True)
dataset = PreprocessedMSADataset(csv_file=config['csv_file'], max_len=config['seq_len'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
print('1', flush=True)

# Initialize the model
model = MSAModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                 num_heads=config['num_heads'], num_layers=config['num_layers'])

# Move the model to the GPUs and use DataParallel
model = nn.DataParallel(model)
model = model.to('cuda')  # Send the model to the available GPUs

print('2', flush=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}", flush=True)

# Define loss function, optimizer, and gradient scaler
criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
scaler = GradScaler()  # Automatically detects if CUDA is available

# Training loop
for epoch in range(config['num_epochs']):
    print(f'Start epoch {epoch}', flush=True)
    model.train()
    total_loss = 0

    for seq1, seq2, tgt1, tgt2 in dataloader:
        optimizer.zero_grad()

        # Move data to the current device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq1, seq2, tgt1, tgt2 = seq1.to(device), seq2.to(device), tgt1.to(device), tgt2.to(device)

        # Forward pass through the model
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output_seq1, output_seq2 = model(seq1, seq2)
            # Create masks to ignore padding and replace padding with 0s
            mask1 = tgt1 != -1
            mask2 = tgt2 != -1
            tgt1_padded = torch.where(mask1, tgt1, torch.zeros_like(tgt1)).float()
            tgt2_padded = torch.where(mask2, tgt2, torch.zeros_like(tgt2)).float()

            # Compute loss for both sequences
            loss_seq1 = criterion(output_seq1, tgt1_padded) * mask1.float()
            loss_seq2 = criterion(output_seq2, tgt2_padded) * mask2.float()

            # Sum losses and compute the mean
            loss = (loss_seq1.sum() + loss_seq2.sum()) / (mask1.sum() + mask2.sum())
            total_loss += loss.item()

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}')

print("Training finished!")

# After training is done, export the learned layers
pretrained_layers = {
    'SelfAttention': model.module.MSASelfAttention.state_dict(),
    'CrossAttention': model.module.MSABidirectionalCrossAttention.state_dict(),
}

# Save the layers to a file
torch.save(pretrained_layers, 'pretrained_layers.pth')

