import torch

# Load the state_dict
state_dict = torch.load('best_paratope_model.pth')

# Print each layer and its parameter shape
for layer_name, param in state_dict.items():
    print(f"Layer: {layer_name} | Shape: {param.shape}")
