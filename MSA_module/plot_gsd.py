import matplotlib.pyplot as plt

# Data
x = [0, 1, 2, 3]
y1 = [0.751, 0.769, 0.779, 0.779]  # AUC-PR of paratope prediction
y2 = [0.472, 0.473, 0.484, 0.490]  # AUC-PR of epitope prediction

# Create two subplots: top for paratope prediction, bottom for epitope prediction
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), sharex=True)

# Plot for paratope prediction (y1)
ax1.plot(x, y1, marker='o', linestyle='-', color='blue', label='Paratope AUC-PR')
max_y1 = max(y1)
max_idx_y1 = y1.index(max_y1)
ax1.scatter(x[max_idx_y1], max_y1, color='red', s=100, zorder=3, label='Max Paratope AUC-PR')
ax1.annotate(
    f'Max: {max_y1:.3f}',
    xy=(x[max_idx_y1], max_y1),
    xytext=(x[max_idx_y1] - 0.5, max_y1 - 0.01),
    textcoords='data',
    arrowprops=dict(facecolor='red', arrowstyle='->'),
    fontsize=10,
    color='red'
)
ax1.set_ylabel('AUC-PR', fontsize=12)
ax1.set_title('Paratope Prediction AUC-PR', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend()

# Plot for epitope prediction (y2)
ax2.plot(x, y2, marker='o', linestyle='-', color='green', label='Epitope AUC-PR')
max_y2 = max(y2)
max_idx_y2 = y2.index(max_y2)
ax2.scatter(x[max_idx_y2], max_y2, color='red', s=100, zorder=3, label='Max Epitope AUC-PR')
ax2.annotate(
    f'Max: {max_y2:.3f}',
    xy=(x[max_idx_y2], max_y2),
    xytext=(x[max_idx_y2] - 0.5, max_y2 - 0.01),
    textcoords='data',
    arrowprops=dict(facecolor='red', arrowstyle='->'),
    fontsize=10,
    color='red'
)
ax2.set_xlabel('Step of CTSR', fontsize=12)
ax2.set_ylabel('AUC-PR', fontsize=12)
ax2.set_title('Epitope Prediction AUC-PR', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend()

# Set x-axis ticks to the original integer values to maintain spacing
ax2.set_xticks(x)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

