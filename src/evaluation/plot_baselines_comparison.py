"""Figures de thèse : comparaison Deep Learning vs SVM vs Random Forest."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12})


def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


# -- Model registry: names, result paths, colors --

DL_MODELS = [
    "SmallRFNet", "RFResNet", "RFTransformer", "RFEfficientNet",
    "RFAST", "RFConformer", "RFCNN1D",
]

DL_KEYS = {
    "SmallRFNet": "smallrf",
    "RFResNet": "resnet",
    "RFTransformer": "transformer",
    "RFEfficientNet": "efficientnet",
    "RFAST": "ast",
    "RFConformer": "conformer",
    "RFCNN1D": "cnn1d",
}

BASELINE_METHODS = ["SVM", "Random Forest"]
ALL_METHODS_BINARY = BASELINE_METHODS + [m for m in DL_MODELS if m != "RFCNN1D"]  # CNN1D only for DroneRF
ALL_METHODS_MULTI = BASELINE_METHODS + DL_MODELS

COLORS = {
    "SVM": "#FF7043",
    "Random Forest": "#42A5F5",
    "SmallRFNet": "#AB47BC",
    "RFResNet": "#66BB6A",
    "RFTransformer": "#FFA726",
    "RFEfficientNet": "#26C6DA",
    "RFAST": "#EC407A",
    "RFConformer": "#8D6E63",
    "RFCNN1D": "#78909C",
}


def _dl_result_path(ds_key, model_key, task):
    """Return path to DL result JSON."""
    return f"outputs/{ds_key}_{model_key}_{task}/results.json"


def _load_baselines(ds_key, task):
    """Load SVM and RF results for a dataset/task."""
    data = {}
    for method, fname in [("SVM", "svm_results.json"), ("Random Forest", "random_forest_results.json")]:
        r = load_json(f"outputs/baselines_{ds_key}_{task}/{fname}")
        if r:
            data[method] = {
                "accuracy": r["test"]["accuracy"],
                "macro_f1": r["test"]["macro_f1"],
            }
    return data


def _load_dl(ds_key, task, models=None):
    """Load DL model results for a dataset/task."""
    if models is None:
        models = DL_MODELS
    data = {}
    for model_name in models:
        model_key = DL_KEYS[model_name]
        r = load_json(_dl_result_path(ds_key, model_key, task))
        if r:
            data[model_name] = {
                "accuracy": r.get("accuracy", 0),
                "macro_f1": r.get("macro_f1", 0),
            }
    return data


def plot_binary_comparison(output_dir):
    """Bar chart: all DL models vs SVM vs RF for binary classification."""

    datasets = ["DroneRF", "CageDroneRF"]
    ds_keys = ["dronerf", "cagedronerf"]

    # DroneRF binary includes CNN1D, CageDroneRF does not
    dronerf_methods = BASELINE_METHODS + DL_MODELS
    cage_methods = [m for m in dronerf_methods if m != "RFCNN1D"]
    all_methods = dronerf_methods  # superset

    data = {}
    for ds_key, ds_label in zip(ds_keys, datasets):
        data[ds_label] = {}
        data[ds_label].update(_load_baselines(ds_key, "binary"))
        available_models = DL_MODELS if ds_key == "dronerf" else [m for m in DL_MODELS if m != "RFCNN1D"]
        data[ds_label].update(_load_dl(ds_key, "binary", available_models))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, metric in enumerate(["accuracy", "macro_f1"]):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        n = len(all_methods)
        width = 0.8 / n

        for i, method in enumerate(all_methods):
            vals = []
            for ds in datasets:
                if ds in data and method in data[ds]:
                    vals.append(data[ds][method][metric])
                else:
                    vals.append(0)

            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                         label=method, color=COLORS[method], edgecolor="black",
                         alpha=0.85)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                            f"{h:.3f}", ha="center", fontsize=7, fontweight="bold",
                            rotation=90, va="bottom")

        metric_label = "Accuracy" if metric == "accuracy" else "Macro-F1"
        ax.set_xlabel("Dataset", fontsize=13)
        ax.set_ylabel(metric_label, fontsize=13)
        ax.set_title(f"Classification binaire — {metric_label}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=12)
        ax.set_ylim(0.82, 1.05)
        ax.legend(fontsize=8, ncol=3, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "baselines_binary_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'baselines_binary_comparison.png'}")


def plot_multiclass_comparison(output_dir):
    """Bar chart: all DL models vs SVM vs RF for multiclass classification."""

    datasets = ["DroneRF\n(4 classes)", "CageDroneRF\n(27 classes)", "RFUAV\n(37 classes)"]
    ds_keys = ["dronerf", "cagedronerf", "rfuav"]

    # CNN1D only available for DroneRF
    data = {}
    for ds_key, ds_label in zip(ds_keys, datasets):
        data[ds_label] = {}
        data[ds_label].update(_load_baselines(ds_key, "multiclass"))
        available_models = DL_MODELS if ds_key == "dronerf" else [m for m in DL_MODELS if m != "RFCNN1D"]
        data[ds_label].update(_load_dl(ds_key, "multiclass", available_models))

    all_methods = BASELINE_METHODS + DL_MODELS

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax_idx, metric in enumerate(["accuracy", "macro_f1"]):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        n = len(all_methods)
        width = 0.8 / n

        for i, method in enumerate(all_methods):
            vals = []
            for ds in datasets:
                if ds in data and method in data[ds]:
                    vals.append(data[ds][method][metric])
                else:
                    vals.append(0)

            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                         label=method, color=COLORS[method], edgecolor="black",
                         alpha=0.85)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                            f"{h:.3f}", ha="center", fontsize=6, fontweight="bold",
                            rotation=90, va="bottom")

        metric_label = "Accuracy" if metric == "accuracy" else "Macro-F1"
        ax.set_xlabel("Dataset", fontsize=13)
        ax.set_ylabel(metric_label, fontsize=13)
        ax.set_title(f"Classification multiclasse — {metric_label}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylim(0.65, 1.08)
        ax.legend(fontsize=8, ncol=3, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "baselines_multiclass_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'baselines_multiclass_comparison.png'}")


def plot_dl_advantage(output_dir):
    """Show F1 gain of best DL model vs best baseline per dataset and task."""

    # For each config: find best DL model dynamically
    configs = [
        ("DroneRF\nbinary", "dronerf", "binary"),
        ("CageDroneRF\nbinary", "cagedronerf", "binary"),
        ("DroneRF\nmulticlass", "dronerf", "multiclass"),
        ("CageDroneRF\nmulticlass", "cagedronerf", "multiclass"),
        ("RFUAV\nmulticlass", "rfuav", "multiclass"),
    ]

    labels = []
    baseline_f1 = []
    dl_f1 = []
    best_dl_names = []

    for label, ds_key, task in configs:
        # Best baseline F1
        svm_r = load_json(f"outputs/baselines_{ds_key}_{task}/svm_results.json")
        rf_r = load_json(f"outputs/baselines_{ds_key}_{task}/random_forest_results.json")
        best_bl = 0
        if svm_r:
            best_bl = max(best_bl, svm_r["test"]["macro_f1"])
        if rf_r:
            best_bl = max(best_bl, rf_r["test"]["macro_f1"])

        # Best DL model F1
        available_models = DL_MODELS if ds_key == "dronerf" else [m for m in DL_MODELS if m != "RFCNN1D"]
        best_dl_f1 = 0
        best_name = ""
        for model_name in available_models:
            model_key = DL_KEYS[model_name]
            r = load_json(_dl_result_path(ds_key, model_key, task))
            if r:
                f1 = r.get("macro_f1", 0)
                if f1 > best_dl_f1:
                    best_dl_f1 = f1
                    best_name = model_name

        if best_bl > 0 and best_dl_f1 > 0:
            labels.append(label)
            baseline_f1.append(best_bl)
            dl_f1.append(best_dl_f1)
            best_dl_names.append(best_name)

    gains = [dl - bl for dl, bl in zip(dl_f1, baseline_f1)]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, baseline_f1, width, label="Meilleure baseline (SVM/RF)",
                   color="#FF7043", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, dl_f1, width, label="Meilleur DL",
                   color="#66BB6A", edgecolor="black", alpha=0.85)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    for i, bar in enumerate(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}\n({best_dl_names[i]})", ha="center", fontsize=8, fontweight="bold")

    # Annotate F1 gains
    for i, gain in enumerate(gains):
        color = "#2E7D32" if gain > 0 else "#C62828"
        sign = "+" if gain > 0 else ""
        ax.annotate(f"{sign}{gain:.3f}",
                    xy=(x[i] + width / 2, dl_f1[i]),
                    xytext=(x[i] + 0.55, min(dl_f1[i] + 0.04, 1.02)),
                    fontsize=10, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax.set_xlabel("Dataset × Tâche", fontsize=13)
    ax.set_ylabel("Macro-F1", fontsize=13)
    ax.set_title("Gain du meilleur modèle Deep Learning par rapport aux baselines", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.7, 1.08)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "baselines_dl_advantage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'baselines_dl_advantage.png'}")


def main():
    output_dir = "outputs/thesis_figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_binary_comparison(output_dir)
    plot_multiclass_comparison(output_dir)
    plot_dl_advantage(output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
