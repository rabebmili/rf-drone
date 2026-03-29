"""Generate thesis-ready comparison figures: Deep Learning vs SVM vs Random Forest."""

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


def plot_binary_comparison(output_dir):
    """Bar chart: DL vs SVM vs RF for binary classification."""

    datasets = ["DroneRF", "CageDroneRF"]
    methods = ["SVM", "Random Forest", "RFResNet"]
    colors = ["#FF7043", "#42A5F5", "#66BB6A"]

    # Collecter les résultats
    data = {}

    # SVM binaire
    for ds_key, ds_label in [("dronerf", "DroneRF"), ("cagedronerf", "CageDroneRF")]:
        r = load_json(f"outputs/baselines_{ds_key}_binary/svm_results.json")
        if r:
            data.setdefault(ds_label, {})["SVM"] = {
                "accuracy": r["test"]["accuracy"],
                "macro_f1": r["test"]["macro_f1"],
            }

    # RF binaire
    for ds_key, ds_label in [("dronerf", "DroneRF"), ("cagedronerf", "CageDroneRF")]:
        r = load_json(f"outputs/baselines_{ds_key}_binary/random_forest_results.json")
        if r:
            data.setdefault(ds_label, {})["Random Forest"] = {
                "accuracy": r["test"]["accuracy"],
                "macro_f1": r["test"]["macro_f1"],
            }

    # DL binaire (RFResNet — meilleur modèle binaire)
    for ds_key, ds_label, path in [
        ("dronerf", "DroneRF", "outputs/resnet_binary/results.json"),
        ("cagedronerf", "CageDroneRF", "outputs/cagedronerf_resnet_binary/results.json"),
    ]:
        r = load_json(path)
        if r:
            data.setdefault(ds_label, {})["RFResNet"] = {
                "accuracy": r.get("accuracy", 0),
                "macro_f1": r.get("macro_f1", 0),
            }

    # Tracer
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, metric in enumerate(["accuracy", "macro_f1"]):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.25

        for i, method in enumerate(methods):
            vals = []
            for ds in datasets:
                if ds in data and method in data[ds]:
                    vals.append(data[ds][method][metric])
                else:
                    vals.append(0)

            bars = ax.bar(x + i * width, vals, width, label=method,
                         color=colors[i], edgecolor="black", alpha=0.85)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                            f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

        metric_label = "Accuracy" if metric == "accuracy" else "Macro-F1"
        ax.set_xlabel("Dataset", fontsize=13)
        ax.set_ylabel(metric_label, fontsize=13)
        ax.set_title(f"Classification binaire — {metric_label}", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, fontsize=12)
        ax.set_ylim(0.85, 1.02)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "baselines_binary_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'baselines_binary_comparison.png'}")


def plot_multiclass_comparison(output_dir):
    """Bar chart: DL vs SVM vs RF for multiclass classification."""

    datasets = ["DroneRF\n(4 classes)", "CageDroneRF\n(27 classes)", "RFUAV\n(37 classes)"]
    ds_keys = ["dronerf", "cagedronerf", "rfuav"]
    methods = ["SVM", "Random Forest", "RFResNet"]
    colors = ["#FF7043", "#42A5F5", "#66BB6A"]

    data = {ds: {} for ds in datasets}

    # SVM multiclasse
    for ds_key, ds_label in zip(ds_keys, datasets):
        r = load_json(f"outputs/baselines_{ds_key}_multiclass/svm_results.json")
        if r:
            data[ds_label]["SVM"] = {
                "accuracy": r["test"]["accuracy"],
                "macro_f1": r["test"]["macro_f1"],
            }

    # RF multiclasse
    for ds_key, ds_label in zip(ds_keys, datasets):
        r = load_json(f"outputs/baselines_{ds_key}_multiclass/random_forest_results.json")
        if r:
            data[ds_label]["Random Forest"] = {
                "accuracy": r["test"]["accuracy"],
                "macro_f1": r["test"]["macro_f1"],
            }

    # DL multiclasse (RFResNet)
    dl_paths = [
        "outputs/resnet_multiclass/results.json",
        "outputs/cagedronerf_resnet_multiclass/results.json",
        "outputs/rfuav_resnet_multiclass/results.json",
    ]
    for ds_label, path in zip(datasets, dl_paths):
        r = load_json(path)
        if r:
            data[ds_label]["RFResNet"] = {
                "accuracy": r.get("accuracy", 0),
                "macro_f1": r.get("macro_f1", 0),
            }

    # Tracer
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, metric in enumerate(["accuracy", "macro_f1"]):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.25

        for i, method in enumerate(methods):
            vals = []
            for ds in datasets:
                if ds in data and method in data[ds]:
                    vals.append(data[ds][method][metric])
                else:
                    vals.append(0)

            bars = ax.bar(x + i * width, vals, width, label=method,
                         color=colors[i], edgecolor="black", alpha=0.85)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", fontsize=8, fontweight="bold")

        metric_label = "Accuracy" if metric == "accuracy" else "Macro-F1"
        ax.set_xlabel("Dataset", fontsize=13)
        ax.set_ylabel(metric_label, fontsize=13)
        ax.set_title(f"Classification multiclasse — {metric_label}", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylim(0.65, 1.05)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "baselines_multiclass_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'baselines_multiclass_comparison.png'}")


def plot_dl_advantage(output_dir):
    """Show DL F1 gain over best baseline per dataset x task."""

    configs = [
        ("DroneRF\nbinary", "dronerf", "binary", "outputs/resnet_binary/results.json"),
        ("CageDroneRF\nbinary", "cagedronerf", "binary", "outputs/cagedronerf_resnet_binary/results.json"),
        ("DroneRF\nmulticlass", "dronerf", "multiclass", "outputs/resnet_multiclass/results.json"),
        ("CageDroneRF\nmulticlass", "cagedronerf", "multiclass", "outputs/cagedronerf_resnet_multiclass/results.json"),
        ("RFUAV\nmulticlass", "rfuav", "multiclass", "outputs/rfuav_resnet_multiclass/results.json"),
    ]

    labels = []
    baseline_f1 = []
    dl_f1 = []

    for label, ds_key, task, dl_path in configs:
        # Meilleur F1 de la baseline
        svm_r = load_json(f"outputs/baselines_{ds_key}_{task}/svm_results.json")
        rf_r = load_json(f"outputs/baselines_{ds_key}_{task}/random_forest_results.json")
        best_bl = 0
        if svm_r:
            best_bl = max(best_bl, svm_r["test"]["macro_f1"])
        if rf_r:
            best_bl = max(best_bl, rf_r["test"]["macro_f1"])

        # F1 du DL
        dl_r = load_json(dl_path)
        dl_val = dl_r.get("macro_f1", 0) if dl_r else 0

        if best_bl > 0 and dl_val > 0:
            labels.append(label)
            baseline_f1.append(best_bl)
            dl_f1.append(dl_val)

    gains = [dl - bl for dl, bl in zip(dl_f1, baseline_f1)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, baseline_f1, width, label="Meilleure baseline (SVM/RF)",
                   color="#FF7043", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, dl_f1, width, label="RFResNet (Deep Learning)",
                   color="#66BB6A", edgecolor="black", alpha=0.85)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Annoter les gains
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
    ax.set_title("Gain du Deep Learning par rapport aux baselines", fontsize=14, fontweight="bold")
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
