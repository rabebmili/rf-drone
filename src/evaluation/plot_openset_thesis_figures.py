"""
Generate three thesis figures for the open-set detection chapter.

Figure 1 — Score distribution composite:
    MSP / Energy / OpenMax score histograms for the best-performing model
    (AST, DroneRF, holdout Bepop Drone) — three subplots side-by-side.

Figure 2 — AUROC bar chart (DroneRF):
    4 methods × 7 models, averaged over holdout classes.
    Shows which model × method combination performs best.

Figure 3 — AUROC heatmap (methods × datasets):
    Averaged over all models and holdout classes per dataset.
    Shows generalisability across the three evaluation corpora.

Outputs are saved to:
    outputs/thesis_figures/openset/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "outputs/multiclass_evaluation/openset_multiclass"
OUT  = "outputs/thesis_figures/openset"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["DroneRF", "CageDroneRF", "RFUAV"]
METHODS  = ["MSP", "Energy", "Mahalanobis", "OpenMax"]
METHOD_LABELS = ["MSP", "Energy", "Mahalanobis", "OpenMax"]
METHOD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

PRETTY = {
    "smallrf":    "SmallRF",
    "resnet":     "ResNet",
    "transformer":"Transformer",
    "efficientnet":"EfficientNet",
    "ast":        "AST",
    "conformer":  "Conformer",
    "cnn1d":      "CNN-1D",
}

# ---------------------------------------------------------------------------
# Per-dataset summaries — loaded lazily inside main() to avoid import-time I/O
# ---------------------------------------------------------------------------
data = {}  # populated by main() before calling the figure functions


# ---------------------------------------------------------------------------
# Figure 1 — Score distribution composite
# ---------------------------------------------------------------------------
# Use AST on DroneRF, holdout Bepop Drone (all three method PNGs exist here)
def make_distribution_composite():
    model_dir = os.path.join(BASE, "DroneRF", "ast", "holdout_Bepop Drone")
    png_files = {
        "MSP":     os.path.join(model_dir, "msp_distribution.png"),
        "Energy":  os.path.join(model_dir, "energy_distribution.png"),
        "OpenMax": os.path.join(model_dir, "openmax_distribution.png"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (method, path) in zip(axes, png_files.items()):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{method} Score Distribution", fontsize=12, fontweight="bold", pad=6)

    fig.suptitle(
        "Open-Set Score Distributions — AST on DroneRF (OOD: Bepop Drone)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig1_score_distributions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — AUROC bar chart: methods × models on DroneRF
# ---------------------------------------------------------------------------
def make_auroc_bar_chart():
    ds_data = data["DroneRF"]
    models = list(ds_data.keys())           # preserves insertion order

    # avg AUROC per (model, method) over all holdout classes
    auroc = {}   # model -> {method -> float}
    for model in models:
        auroc[model] = {}
        holdouts = list(ds_data[model].keys())
        for method in METHODS:
            vals = []
            for hc in holdouts:
                if method in ds_data[model][hc]:
                    vals.append(ds_data[model][hc][method]["auroc"])
            auroc[model][method] = np.mean(vals) if vals else np.nan

    x = np.arange(len(models))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (method, color, label) in enumerate(zip(METHODS, METHOD_COLORS, METHOD_LABELS)):
        vals = [auroc[m][method] for m in models]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        # annotate values on top
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in models], fontsize=11)
    ax.set_ylabel("AUROC (avg. over holdout classes)", fontsize=11)
    ax.set_title("Open-Set Detection AUROC — DroneRF", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random baseline")
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = os.path.join(OUT, "fig2_auroc_bar_chart_dronerf.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — AUROC heatmap: methods × datasets
# ---------------------------------------------------------------------------
def make_auroc_heatmap():
    # Build matrix: rows = methods, cols = datasets
    # Each cell = mean AUROC over all models × holdout classes
    matrix = np.zeros((len(METHODS), len(DATASETS)))
    annot  = np.empty((len(METHODS), len(DATASETS)), dtype=object)

    for j, ds in enumerate(DATASETS):
        ds_data = data[ds]
        for i, method in enumerate(METHODS):
            vals = []
            for model in ds_data:
                for hc in ds_data[model]:
                    if method in ds_data[model][hc]:
                        vals.append(ds_data[model][hc][method]["auroc"])
            mean_val = np.mean(vals) if vals else np.nan
            matrix[i, j] = mean_val
            annot[i, j]  = f"{mean_val:.3f}" if not np.isnan(mean_val) else "—"

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.3, vmax=0.9, aspect="auto")

    # Axes labels
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, fontsize=11)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHOD_LABELS, fontsize=11)

    # Annotate cells
    for i in range(len(METHODS)):
        for j in range(len(DATASETS)):
            val = matrix[i, j]
            text_color = "black" if 0.4 < val < 0.8 else "white"
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC", fontsize=10)
    ax.set_title("Open-Set AUROC — Methods × Datasets\n(averaged over all models and holdout classes)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(OUT, "fig3_auroc_heatmap_methods_datasets.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
def main():
    global data
    print("Generating open-set thesis figures...")
    for ds in DATASETS:
        path = os.path.join(BASE, ds, "openset_summary.json")
        with open(path) as f:
            data[ds] = json.load(f)
    make_distribution_composite()
    make_auroc_bar_chart()
    make_auroc_heatmap()
    print("Done. All figures saved to:", OUT)


if __name__ == "__main__":
    main()
