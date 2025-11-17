import torch

# Load the checkpoint file (using CPU)
state_dict = torch.load('isicIHDreg_l1_g12_i6_do0.1_dpr0.1_fold5.pth', map_location=torch.device('cpu'))

# Get the keys 
keys = list(state_dict.keys())

# Print the keys
print(keys)

num_params = sum(p.numel() for p in state_dict.values())
print("Total parameters:", num_params)
