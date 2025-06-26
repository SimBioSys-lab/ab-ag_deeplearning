import matplotlib.pyplot as plt
import numpy as np

def add_bar_labels_inside(rects, ax, fmt="{:.3f}", offset_ratio=0.05, color="black"):
    """
    Place a text label inside each bar near the bottom.

    Args:
        rects (list): List of matplotlib.patches.Rectangle objects (bars).
        ax (matplotlib.axes.Axes): The Axes object to draw the text on.
        fmt (str): Format string for bar labels.
        offset_ratio (float): Fraction of bar height to place the label above the bottom.
        color (str): Color of the text.
    """
    for rect in rects:
        height = rect.get_height()
        label_y = height * offset_ratio  # label position from bottom
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, label_y),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    rotation=90,
                    color=color)

def plot_grouped_bars(ax, data, metrics, variants, bar_width=0.08, group_gap_scale=0.8, y_max=1.0, title="", label_color="black"):
    """
    Plot grouped bars for a given dataset on the provided Axes object.
    
    Args:
        ax (matplotlib.axes.Axes): The Axes object to plot on.
        data (np.ndarray): Shape (n_metrics, n_variants).
        metrics (list): List of metric names (x-axis labels).
        variants (list): List of variant/model names (legend labels).
        bar_width (float): Width of each bar.
        group_gap_scale (float): Factor to scale the gap between metric groups (<1 reduces gap).
        y_max (float): Upper limit for the y-axis.
        title (str): Title for the subplot.
        label_color (str): Color for the text labels.
    """
    n_metrics, n_variants = data.shape
    # Generate x positions for each metric group, then scale to reduce the gap.
    x = np.arange(n_metrics) * group_gap_scale

    for i in range(n_variants):
        rects = ax.bar(
            x + i * bar_width,
            data[:, i],
            bar_width,
            label=variants[i]
        )
        add_bar_labels_inside(rects, ax, color=label_color)
    
    ax.set_title(title)
    # Center xticks relative to the group
    ax.set_xticks(x + (n_variants - 1) * bar_width / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, y_max])
    ax.set_ylabel("Performance")

def main():
    # Define metrics and variants
    metrics = ["AUC", "AUPR"]
    variants = [
        "MGI(w/o Feeding Forward Layers)",
        "MGI(w/o DyM Layers)",
        "MGI(w/o GNN Module)",
        "MGI(w/o Interactive Module)",
        "MGI"
    ]

    # Example data for Paratope and Epitope
    paratope_data = np.array([
        [0.961, 0.830, 0.822, 0.964, 0.966],  # AUC
        [0.700, 0.679, 0.276, 0.722, 0.726]   # AUPR
    ])
    epitope_data = np.array([
        [0.941, 0.800, 0.687, 0.899, 0.940],  # AUC
        [0.536, 0.408, 0.070, 0.365, 0.545]   # AUPR
    ])

    # Create two subplots: one for Paratope and one for Epitope
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Paratope data with narrower bars and reduced group gap
    plot_grouped_bars(ax1, paratope_data, metrics, variants, bar_width=0.08, group_gap_scale=0.8, y_max=1.0,
                      title="Paratope", label_color="white")  # Use white if bars are dark
    ax1.legend(ncol=3)

    # Plot Epitope data with narrower bars and reduced group gap
    plot_grouped_bars(ax2, epitope_data, metrics, variants, bar_width=0.08, group_gap_scale=0.8, y_max=1.0,
                      title="Epitope", label_color="black")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

