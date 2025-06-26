import torch

# Load the checkpoint file (using CPU)
state_dict = torch.load('PPI_model_l0_g20_i8_do0.15_dpr0.15_lr0.0002_fold1_core.pth', map_location=torch.device('cpu'))

# Get the keys 
keys = list(state_dict.keys())

# Print the keys
print(keys)
