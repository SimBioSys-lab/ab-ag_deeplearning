import torch

def modify_key_names(file_list, save_modified=True):
    """
    Modify key names in .pth files. Rename `cg_model.fc_**` back to `fc_**`.

    Args:
        file_list (str): Path to the file containing the list of .pth filenames.
        save_modified (bool): Whether to save the modified state dict back to the original files.
    """
    with open(file_list, 'r') as f:
        files = f.readlines()

    for file in files:
        file = file.strip()  # Remove any trailing whitespace or newlines
        if not file.endswith('.pth'):
            print(f"Skipping {file}, not a .pth file.")
            continue

        print(f"Processing file: {file}")
        try:
            # Load the state dictionary
            state_dict = torch.load(file, map_location=torch.device('cpu'))

            # Modify key names
            updated_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module..") and not key.startswith("module.fc_"):
                    new_key = key.replace("module..", "module.", 1)
                else:
                    # Keep other keys unchanged
                    new_key = key
                updated_state_dict[new_key] = value

            # Save the modified state dictionary back to the same file
            if save_modified:
                torch.save(updated_state_dict, file)
                print(f"Updated and saved: {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Path to the file containing the list of .pth files
model_files = "model_files"

# Run the modification script
modify_key_names(model_files)

