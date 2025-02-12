import torch

# Define input and output paths
input_path = "itfssitfsaitfssitfsaitfssitfmodel_fold5_l1_g2_i1_dp0.1.pth"
output_path = "itfssitfsaitfssitfsaitfssitfmodel_fold5_l1_g2_i1_dp0.1_core.pth"

# Load the model's state dictionary
state_dict = torch.load(input_path, map_location=torch.device('cpu'))

# Extract parameters containing "cg_model"
cg_model_state_dict = {key: value for key, value in state_dict.items() if "mc_model" in key}

# Save the filtered parameters to a new file
torch.save(cg_model_state_dict, output_path)

print(f"Filtered parameters containing 'cg_model' saved to {output_path}")

