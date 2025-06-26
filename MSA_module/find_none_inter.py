import numpy as np

# Replace 'your_file.npz' with the path to your NPZ file.
npz_file = np.load('padded_train_interfaces4.5_2400.npz')

# Iterate through all keys in the NPZ file
for key in npz_file.files:
    data = npz_file[key]
    # Check if all elements in the array are smaller than 1.
    if np.all(data < 1):
        print(key)

