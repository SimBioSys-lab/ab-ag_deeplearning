import matplotlib.pyplot as plt
import numpy as np

def annotate_points(ax, x, y, fmt="{:.3f}", fontsize=9):
    """Annotate each point slightly above its marker."""
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), (xi, yi),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize)

def main():
    # Variants (x-axis)
    variants = [
        "MGI(w/o DyM Layers)",
        "MGI(w/o MSA Module)",
        "MGI(w/o GNN Module)",
        "MGI(w/o Attention Module)",
        "MGI"
    ]
    
    # Reduce spacing between x-axis groups
    x = np.arange(len(variants)) * 0.7

    # Data (metrics x variants)
    paratope_data = np.array([
        [0.982, 0.982, 0.964, 0.980, 0.982],  # AUC
        [0.753, 0.747, 0.584, 0.726, 0.751]   # AUPR
    ])
    epitope_data = np.array([
        [0.809, 0.801, 0.766, 0.798, 0.812],  # AUC
        [0.466, 0.449, 0.409, 0.454, 0.472]   # AUPR
    ])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # === AUC subplot ===
    ax1.plot(x, paratope_data[0], marker='o', linewidth=2.5, markersize=8,
             color="tab:blue", label="Paratope AUC")
    annotate_points(ax1, x, paratope_data[0])
    ax1.plot(x, epitope_data[0], marker='o', linewidth=2.5, markersize=8,
             color="tab:green", label="Epitope AUC")
    annotate_points(ax1, x, epitope_data[0])

    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.75, 1.0)   # zoom for AUC
    ax1.set_title("MGI Ablations: AUC")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # === AUPR subplot ===
    ax2.plot(x, paratope_data[1], marker='s', linestyle='--',
             linewidth=2.5, markersize=8,
             color="tab:orange", label="Paratope AUPR")
    annotate_points(ax2, x, paratope_data[1])
    ax2.plot(x, epitope_data[1], marker='s', linestyle='--',
             linewidth=2.5, markersize=8,
             color="tab:red", label="Epitope AUPR")
    annotate_points(ax2, x, epitope_data[1])

    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=0, ha="center")
    ax2.set_ylabel("AUPR")
    ax2.set_ylim(0.35, 0.8)   # zoom for AUPR
    ax2.set_title("MGI Ablations: AUPR")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

