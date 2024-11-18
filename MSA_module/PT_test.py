import torch
from torch.utils.data import DataLoader
from Dataloader_SS_SASA import SequenceParatopeDataset  # Make sure this matches your dataset import
from Models import ParatopeModel  # Make sure this matches your model import

# Configuration for the test
test_config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_test_1200.npz',  # Path to your test sequence file
    'pt_file': 'pt_test_data.csv',  # Path to your test target file
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 1,
    'num_classes': 2
}

# Load the test dataset
test_dataset = SequenceParatopeDataset(sequence_file=test_config['sequence_file'], pt_file=test_config['pt_file'], max_len=test_config['seq_len'])
test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False)

# Load the best model state
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParatopeModel(vocab_size=test_config['vocab_size'], seq_len=test_config['seq_len'], embed_dim=test_config['embed_dim'],
                      num_heads=test_config['num_heads'], num_layers=test_config['num_layers'], num_classes=test_config['num_classes'])

# Load state_dict and handle DataParallel prefix if necessary
checkpoint = torch.load('best_paratope_model.pth', map_location=device)
if 'module.' in list(checkpoint.keys())[0]:  # Check if keys are prefixed with 'module.'
    checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}  # Remove 'module.' prefix

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# Initialize variables for tracking accuracy
correct = 0
total = 0

# Testing loop
with torch.no_grad():
    for seq, tgt in test_loader:
        seq, tgt = seq.to(device), tgt.to(device)

        # Forward pass
        outputs = model(seq)
        _, predicted = torch.max(outputs, dim=2)  # Choose class with highest score in each position

        # Remove padding positions from accuracy calculation
        mask = tgt >= 0  # Assuming -1 is the padding index in your target tensor
        correct += (predicted[mask] == tgt[mask]).sum().item()
        total += mask.sum().item()

# Calculate accuracy
accuracy = correct / total if total > 0 else 0
print(f'Test Accuracy: {accuracy * 100:.2f}%')

