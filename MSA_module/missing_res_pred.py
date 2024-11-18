import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader_mask import PreprocessedMSADataset  # Assuming your dataset class is in 'DataLoader_mask.py'
from Models import MRPModel  # Assuming your MRPModel class is defined in 'Models'
from torch.amp import autocast, GradScaler

# Hyperparameters and file paths
config = {
    'batch_size': 8,
    'csv_file': 'preprocessed_data_mask_1200.csv',
    'seq_len': 1200,
    'vocab_size': 22,  # 20 amino acids + 1 unknown token + 1 padding token
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 1,
    'num_epochs': 10,
    'learning_rate': 0.01
}

# Load dataset and dataloader
print('1')
dataset = PreprocessedMSADataset(config['csv_file'], config['seq_len'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
print('2')
# Initialize the model, loss function, and optimizer
model = MRPModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                 num_heads=config['num_heads'], num_layers=config['num_layers'])
print('3')
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler('cuda')

# Training loop
for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0
    # Training loop
    for seq1, seq2, tgt1_masked_one_hot, tgt2_masked_one_hot, mask_pos1 in dataloader:
        optimizer.zero_grad()
        with autocast('cuda'):
            print("mask_pos1", mask_pos1)
            # Forward pass with masked sequences and masked positions
            output1 = model(seq1,  mask_pos1)
            print('output_shape', output1.shape)
            print('target_shape', tgt1_masked_one_hot.shape)
            # Compute loss for the masked positions (now using one-hot targets)
            loss = criterion(output1, tgt1_masked_one_hot)
        
            # Combine losses
            total_loss += loss.item()
        
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()    
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}')
    
print("Training finished!")

# After training is done, export the learned layers
pretrained_layers = {
    'SelfAttention': model.MSASelfAttention.state_dict(),
    'CrossAttention': model.MSABidirectionalCrossAttention.state_dict(),
}

# Save the layers to a file
torch.save(pretrained_layers, 'pretrained_layers_mrp.pth')

## Example prediction usage
#model.eval()  # Set model to evaluation mode
#with torch.no_grad():
#    for seq1, seq2, _, _, _, _ in dataloader:  # Ignore target values during prediction
#        # Randomly select positions to mask in each sequence for prediction
#        mask_pos1 = torch.randint(0, config['seq_len'], (seq1.size(0),))
#        mask_pos2 = torch.randint(0, config['seq_len'], (seq2.size(0),))
#
#        # Mask the sequences
#        seq1_masked = seq1.clone()
#        seq2_masked = seq2.clone()
#        for i in range(seq1.size(0)):
#            seq1_masked[i, mask_pos1[i]] = 0  # Mask with 'PAD'
#            seq2_masked[i, mask_pos2[i]] = 0
#
#        # Predict the masked residues
#        pred_residues1, pred_residues2 = model.predict(seq1_masked, seq2_masked, mask_pos1, mask_pos2)
#        print(f"Predicted residues for seq1: {pred_residues1}")
#        print(f"Predicted residues for seq2: {pred_residues2}")
#
