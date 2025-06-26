import torch

# Load the state_dict
state_dict = torch.load('PPI_model_l0_g20_i8_do0.15_dpr0.15_lr0.0002_fold1.pth')

# Print each layer and its parameter shape
for layer_name, param in state_dict.items():
    print(f"Layer: {layer_name} | Shape: {param.shape}")
