import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


att_mat = np.load('global_maps_ihd_esmtrain.npz')
# Example: Create a dummy array with shape (1, 4, 2400, 2400)
# Replace this with your actual array.
arr = att_mat['1ahw']
#heatmap=arr
### Average over the second dimension (axis=1).
##averaged_arr = np.mean(arr, axis=1)  # Resulting shape is (1, 2400, 2400)
#
### Since the first dimension is 1, we can index it out.
##heatmap = averaged_arr[0]  # Now heatmap has shape (2400, 2400)
#
## Normalize the matrix so that the minimum is 0 and the maximum is 1.
#heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
## Use PowerNorm to enhance contrast.
## A gamma value less than 1 will emphasize lower values (which map to white in 'gray_r'),
## making the dark regions (values near 1) more prominent.
#norm = mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=1)

# Create a figure with a white background.
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
ax.set_facecolor('white')

# Plot the heatmap using a reversed grayscale colormap so that 0 is white and 1 is black.
cax = ax.imshow(arr, cmap='gray_r')

# Add a title and colorbar.
ax.set_title('Contrast-Enhanced Normalized Heatmap', color='black')
plt.colorbar(cax, ax=ax, label='Normalized Value')

plt.show()
