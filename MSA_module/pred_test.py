import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader_att import PreprocessedMSADataset  # Assuming your dataset class is in 'DataLoader_att.py'
from Models import MSAModel  # Assuming your MSAModel class is defined in 'Models'
config = {
    'batch_size': 8,
    'csv_file': 'preprocessed_data_1200.csv',
    'seq_len': 1200,   #1500
    'vocab_size': 22,  # 20 amino acids + 1 unknown token + 1 padding token
    'embed_dim': 32,  #128
    'num_heads': 2,     #8
    'num_layers': 1,
    'num_epochs': 10,
    'learning_rate': 0.01
}

# Load dataset and dataloader
print('0', flush=True)
dataset = PreprocessedMSADataset(config['csv_file'], config['seq_len'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
print('1', flush=True)
# Initialize the model, loss function, and optimizer
model = MSAModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'], 
                 num_heads=config['num_heads'], num_layers=config['num_layers'])
print('2', flush=True)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}")
criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0

    for seq1, seq2, tgt1, tgt2 in dataloader:
        optimizer.zero_grad()

        # Forward pass through the model
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
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}')

print("Training finished!")

# After training is done, export the learned layers
pretrained_layers = {
    'SelfAttention': model.MSASelfAttention.state_dict(),
    'CrossAttention': model.MSABidirectionalCrossAttention.state_dict(),
}

# Save the layers to a file
torch.save(pretrained_layers, 'pretrained_layers.pth')


## Example prediction usage
#model.eval()
#with torch.no_grad():
#    for seq1, seq2, tgt1, tgt2 in dataloader:
#        bool_output1, bool_output2 = model.predict(seq1, seq2)
#        print(bool_output1)  # Boolean output tensor
#        print(bool_output2)
