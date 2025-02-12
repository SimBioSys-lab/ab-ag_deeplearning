import torch

# Load the state dict
pth_file = "mcmodel_fold1_l1_g2_i1_dp0.0.pth"
state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

# Calculate total parameters
total_params = sum(param.numel() for param in state_dict.values())
print(f"Total parameters in .pth file: {total_params}")
