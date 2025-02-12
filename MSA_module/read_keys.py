import torch

# Load the checkpoint file (using CPU)
state_dict = torch.load('mcmodel_fold1_l1_g2_i1_dp0.1.pth', map_location=torch.device('cpu'))

# Get the keys 
keys = list(state_dict.keys())

# Print the keys
print(keys)
