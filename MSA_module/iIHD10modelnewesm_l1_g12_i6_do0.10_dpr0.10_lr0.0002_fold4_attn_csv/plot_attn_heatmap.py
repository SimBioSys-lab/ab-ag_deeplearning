import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser(
        description="Plot antibody–antigen attention heatmap with 3-letter AA codes and standard order."
    )
    ap.add_argument("--matrix", required=True, help="CSV file with 20x20 attention matrix (rows=antigen, cols=antibody)")
    ap.add_argument("--out", required=True, help="Output PNG filename for the heatmap")
    args = ap.parse_args()

    # Standard biochemical order
    aa3_order = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL"
    ]

    # 1-letter → 3-letter conversion (for matching input)
    aa3_map = {
        "L":"LEU","A":"ALA","G":"GLY","V":"VAL","S":"SER","E":"GLU","R":"ARG",
        "T":"THR","I":"ILE","D":"ASP","P":"PRO","K":"LYS","Q":"GLN","N":"ASN",
        "F":"PHE","Y":"TYR","M":"MET","H":"HIS","W":"TRP","C":"CYS"
    }

    # Load matrix
    df = pd.read_csv(args.matrix, index_col=0)

    # Convert to 3-letter AA names
    df.index = [aa3_map.get(a, a) for a in df.index]
    df.columns = [aa3_map.get(a, a) for a in df.columns]

    # Reorder rows and columns to standard order
    df = df.reindex(index=aa3_order, columns=aa3_order)

    # Plot
    plt.figure(figsize=(9,8))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    ax = sns.heatmap(
        df,
        cmap=cmap,
        square=True,
        cbar_kws={"label": "Average Attention"},
        linewidths=0.3,
        linecolor="white"
    )

    # Move antibody axis to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Antibody Residue", fontsize=16, fontweight="bold", labelpad=12)
    ax.set_ylabel("Antigen Residue", fontsize=16, fontweight="bold", labelpad=12)

    # Ticks and style
    plt.xticks(rotation=45, ha="left", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    ax.figure.axes[-1].tick_params(labelsize=10)  # colorbar

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved heatmap → {args.out}")

if __name__ == "__main__":
    main()

