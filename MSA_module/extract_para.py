import torch

# Define input and output paths
input_path = "PTSAPT_l1_g3_d256_h16_bs4x2.pth"
output_path = "PTSAPT_l1_g3_d256_h16_bs4x2_core.pth"

# Load the model's state dictionary
state_dict = torch.load(input_path, map_location=torch.device('cpu'))

# Extract parameters containing "cg_model"
cg_model_state_dict = {key: value for key, value in state_dict.items() if "cg_model" in key}

# Save the filtered parameters to a new file
torch.save(cg_model_state_dict, output_path)

print(f"Filtered parameters containing 'cg_model' saved to {output_path}")

