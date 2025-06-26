import numpy as np

# Replace 'your_file.npz' with the path to your .npz file.
npz_data = np.load('MIPE_tv_interfaces.npz')

# Initialize counters for overall ones and zeros.
total_ones = 0
total_zeros = 0

# Iterate over each key and update the counts.
for key in npz_data:
    arr = npz_data[key]
    total_ones += np.sum(arr == 1)
    total_zeros += np.sum(arr == 0)

# Display the total counts.
print("Overall Count of 1's:", total_ones)
print("Overall Count of 0's:", total_zeros)

# Calculate and display the overall ratio.
if total_zeros == 0:
    print("Overall ratio (1's / 0's): Infinite (division by zero)")
else:
    overall_ratio = total_ones / total_zeros
    print("Overall ratio (1's / 0's):", overall_ratio)

