import torch

# Load the state dict
pth_file = "iParamodel_l0_g20_i8_do0.15_dpr0.15_lr0.0002_fold1.pth"
state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

# Calculate total parameters
total_params = sum(param.numel() for param in state_dict.values())
print(f"Total parameters in .pth file: {total_params}")
