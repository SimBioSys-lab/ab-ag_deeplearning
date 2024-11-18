import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataloader_SS_SASA import SequenceParatopeDataset  # Assuming your dataset class is defined here
from Models import ParatopeModel  # Assuming your model class is defined here
from torch.amp import autocast, GradScaler
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration for the model and training
config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_1200.npz',
    'pt_file': 'cdrs_output.csv',
    'seq_len': 1200,   # Sequence length
    'vocab_size': 22,  # 20 amino acids + 1 unknown token + 1 padding token
    'embed_dim': 128,  # Embedding dimension
    'num_heads': 8,    # Number of attention heads
    'num_layers': 1,   # Number of layers in the model
    'num_classes': 3,  # Number of secondary structure classes
    'num_epochs': 10,  # Number of training epochs
    'learning_rate': 0.01  # Learning rate for optimizer
}

# Load dataset and dataloader
print('Initializing dataset...', flush=True)
dataset = SequenceParatopeDataset(sequence_file=config['sequence_file'], pt_file=config['pt_file'], max_len=config['seq_len'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
print('Dataset loaded successfully.', flush=True)

# Initialize the model
model = ParatopeModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                                num_heads=config['num_heads'], num_layers=config['num_layers'], num_classes=config['num_classes'])

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
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Assuming -1 is used for padding targets
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
            output_seq = model(seq)  # Output shape: [batch_size, seq_len, num_classes]

            # Reshape target and predictions to match the format expected by CrossEntropyLoss
            # Target shape should be [batch_size, seq_len]
            # Prediction shape should be [batch_size * seq_len, num_classes]
            output_seq = output_seq.view(-1, config['num_classes'])  # Flatten the output for cross-entropy
            tgt = tgt.view(-1)  # Flatten the target to match the flattened output

            # Compute loss for secondary structure classification
            loss = criterion(output_seq, tgt)

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
torch.save(pretrained_layers, 'pretrained_PT.pth')
print("Model layers saved successfully.", flush=True)

