"""Figures de thèse pour l'évaluation inter-datasets."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12})


def load_metrics(base_dir):
    # Charger toutes les métriques depuis le répertoire de sortie inter-datasets
    base = Path(base_dir)
    results = {}

    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        for mf in sorted(exp_dir.rglob("*_metrics.json")):
            with open(mf) as f:
                m = json.load(f)
            # Supprimer les NaN
            for k, v in m.items():
                if isinstance(v, float) and np.isnan(v):
                    m[k] = None
            results[str(mf.relative_to(base))] = m

    return results


def plot_cross_dataset_heatmap(output_dir):
    # Heatmap : entraîner sur X, tester sur Y (source unique). RFUAV exclu comme cible
    train_datasets = ["DroneRF", "CageDroneRF", "RFUAV"]
    test_datasets = ["DroneRF", "CageDroneRF"]  # RFUAV excluded: no background
    base = Path("outputs/cross_dataset_enhanced")

    n_train, n_test = len(train_datasets), len(test_datasets)
    acc_matrix = np.zeros((n_train, n_test))
    f1_matrix = np.zeros((n_train, n_test))

    for i, src in enumerate(train_datasets):
        for j, tgt in enumerate(test_datasets):
            mf = base / f"single_{src}" / f"single_{src}_{tgt}_metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                acc_matrix[i, j] = m["accuracy"]
                f1_matrix[i, j] = m["macro_f1"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, matrix, title, fmt in [
        (axes[0], acc_matrix, "Cross-Dataset Accuracy (Binary)", ".2%"),
        (axes[1], f1_matrix, "Cross-Dataset Macro-F1 (Binary)", ".3f"),
    ]:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_test))
        ax.set_yticks(range(n_train))
        ax.set_xticklabels(test_datasets, fontsize=11)
        ax.set_yticklabels(train_datasets, fontsize=11)
        ax.set_xlabel("Tested on", fontsize=13)
        ax.set_ylabel("Trained on", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")

        for ii in range(n_train):
            for jj in range(n_test):
                val = matrix[ii, jj]
                color = "white" if val < 0.5 else "black"
                if fmt == ".2%":
                    text = f"{val:.1%}"
                else:
                    text = f"{val:.3f}"
                ax.text(jj, ii, text, ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "cross_dataset_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'cross_dataset_heatmap.png'}")


def plot_leave_one_out(output_dir):
    # Diagramme en barres : résultats leave-one-dataset-out. RFUAV exclu
    datasets = ["DroneRF", "CageDroneRF"]  # RFUAV excluded: no background
    base = Path("outputs/cross_dataset_enhanced")

    held_out_names = []
    accs = []
    f1s = []

    for held in datasets:
        mf = base / f"leave_out_{held}" / f"leave_out_{held}_{held}_metrics.json"
        if mf.exists():
            with open(mf) as f:
                m = json.load(f)
            held_out_names.append(held)
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])

    x = np.arange(len(held_out_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#2196F3", edgecolor="black")
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro-F1", color="#FF9800", edgecolor="black")

    ax.set_xlabel("Held-out Dataset (unseen during training)", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Leave-One-Dataset-Out: Generalization to Unseen Datasets", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(held_out_names, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random baseline")

    # Ajouter les étiquettes de valeurs
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.1%}", ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "leave_one_out_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'leave_one_out_bar.png'}")


def plot_ablation_comparison(output_dir):
    # Diagramme groupé : unique vs paires vs 3 simples vs 3 équilibrés
    base = Path("outputs/cross_dataset_enhanced")
    datasets = ["DroneRF", "CageDroneRF"]  # RFUAV excluded: no background
    all_datasets = ["DroneRF", "CageDroneRF", "RFUAV"]  # for training combos

    experiments = {
        "Single\n(best)": None,  # will pick best single for each target
        "Pairwise\n(best)": None,
        "All-3\nPlain": "all3_plain",
        "All-3\nBalanced": "all3_balanced",
    }

    # Collecter les données
    data = {ds: {} for ds in datasets}

    for tgt in datasets:
        # Meilleur source unique (recherche parmi les 3 sources d'entraînement)
        best_f1 = 0
        for src in all_datasets:
            mf = base / f"single_{src}" / f"single_{src}_{tgt}_metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                if m["macro_f1"] > best_f1:
                    best_f1 = m["macro_f1"]
        data[tgt]["Single\n(best)"] = best_f1

        # Meilleure paire (toutes les combinaisons de 2 parmi 3)
        best_f1 = 0
        import itertools
        for pair in itertools.combinations(all_datasets, 2):
            pair_name = "+".join(pair)
            mf = base / f"pair_{pair_name}" / f"pair_{pair_name}_{tgt}_metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                if m["macro_f1"] > best_f1:
                    best_f1 = m["macro_f1"]
        data[tgt]["Pairwise\n(best)"] = best_f1

        # Les 3 simple et équilibré
        for label, dirname in [("All-3\nPlain", "all3_plain"), ("All-3\nBalanced", "all3_balanced")]:
            mf = base / dirname / f"{dirname}_{tgt}_metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                data[tgt][label] = m["macro_f1"]

    # Tracer
    exp_labels = ["Single\n(best)", "Pairwise\n(best)", "All-3\nPlain", "All-3\nBalanced"]
    x = np.arange(len(exp_labels))
    width = 0.25
    colors = ["#E53935", "#43A047", "#1E88E5"]

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, ds in enumerate(datasets):
        vals = [data[ds].get(exp, 0) for exp in exp_labels]
        bars = ax.bar(x + i * width, vals, width, label=ds, color=colors[i],
                      edgecolor="black", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Training Configuration", fontsize=13)
    ax.set_ylabel("Macro-F1", fontsize=13)
    ax.set_title("Ablation Study: Effect of Dataset Combination Strategy", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(exp_labels, fontsize=11)
    ax.set_ylim(0.85, 1.05)
    ax.legend(title="Tested on", fontsize=10, title_fontsize=11)

    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "ablation_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'ablation_comparison.png'}")


def plot_finetune_comparison(output_dir):
    # Comparer : entraînement depuis zéro vs pré-entraînement+affinage. RFUAV exclu
    base = Path("outputs/cross_dataset_enhanced")
    datasets = ["DroneRF", "CageDroneRF"]  # RFUAV excluded: no background

    single_f1 = []
    finetune_f1 = []
    labels = []

    for ds in datasets:
        # Source unique (entraînement uniquement sur la cible)
        mf = base / f"single_{ds}" / f"single_{ds}_{ds}_metrics.json"
        if mf.exists():
            with open(mf) as f:
                m = json.load(f)
            single_f1.append(m["macro_f1"])
        else:
            single_f1.append(0)

        # Affine (pré-entraînement sur tous → affinage sur la cible)
        mf = base / "finetune" / f"ft_{ds}" / f"ft_{ds}_{ds}_metrics.json"
        if mf.exists():
            with open(mf) as f:
                m = json.load(f)
            finetune_f1.append(m["macro_f1"])
        else:
            finetune_f1.append(0)

        labels.append(ds)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, single_f1, width, label="Train from scratch\n(target only)",
                   color="#E53935", edgecolor="black")
    bars2 = ax.bar(x + width/2, finetune_f1, width, label="Pretrain (all 3)\n+ fine-tune on target",
                   color="#43A047", edgecolor="black")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Target Dataset", fontsize=13)
    ax.set_ylabel("Macro-F1", fontsize=13)
    ax.set_title("Fine-Tuning: Pretrain on All vs Train from Scratch", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "finetune_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'finetune_comparison.png'}")


def plot_per_dataset_model_comparison(output_dir):
    # Diagramme comparant 3 modèles sur chaque dataset (résultats même-dataset)
    models = ["smallrf", "resnet", "transformer"]
    model_labels = ["SmallRFNet", "RFResNet", "RFTransformer"]
    datasets_tasks = [
        ("DroneRF", "binary", "outputs/dronerf_{model}_binary/results.json"),
        ("DroneRF", "multiclass", "outputs/dronerf_{model}_multiclass/results.json"),
        ("CageDroneRF", "binary", "outputs/cagedronerf_{model}_binary/results.json"),
        ("CageDroneRF", "multiclass", "outputs/cagedronerf_{model}_multiclass/results.json"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax_idx, (ds_name, task, path_template) in enumerate(datasets_tasks):
        ax = axes[ax_idx // 2][ax_idx % 2]
        accs = []
        f1s = []

        for model in models:
            path = Path(path_template.format(model=model))
            if path.exists():
                with open(path) as f:
                    r = json.load(f)
                accs.append(r.get("accuracy", 0))
                f1s.append(r.get("macro_f1", 0))
            else:
                accs.append(0)
                f1s.append(0)

        x = np.arange(len(models))
        width = 0.35
        bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#2196F3", edgecolor="black")
        bars2 = ax.bar(x + width/2, f1s, width, label="Macro-F1", color="#FF9800", edgecolor="black")

        for bar in bars1:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_title(f"{ds_name} — {task}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.set_ylim(0.9, 1.02)
        ax.legend(fontsize=9)
        ax.set_ylabel("Score")

    plt.suptitle("Model Comparison Across Datasets", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "model_comparison_all_datasets.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'model_comparison_all_datasets.png'}")


def plot_domain_shift_summary(output_dir):
    # Résumé : baisse de performance même-dataset vs inter-datasets. RFUAV exclu
    base = Path("outputs/cross_dataset_enhanced")
    datasets = ["DroneRF", "CageDroneRF"]  # RFUAV excluded: no background

    same_ds_f1 = []
    cross_ds_f1_avg = []
    labels = []

    for ds in datasets:
        # F1 même-dataset
        mf = base / f"single_{ds}" / f"single_{ds}_{ds}_metrics.json"
        if mf.exists():
            with open(mf) as f:
                m = json.load(f)
            same_ds_f1.append(m["macro_f1"])
        else:
            same_ds_f1.append(0)

        # F1 inter-datasets (testé sur l'autre dataset valide)
        cross_f1s = []
        for other in datasets:
            if other == ds:
                continue
            mf = base / f"single_{ds}" / f"single_{ds}_{other}_metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                cross_f1s.append(m["macro_f1"])
        cross_ds_f1_avg.append(np.mean(cross_f1s) if cross_f1s else 0)
        labels.append(ds)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, same_ds_f1, width, label="Same-dataset F1",
                   color="#43A047", edgecolor="black")
    bars2 = ax.bar(x + width/2, cross_ds_f1_avg, width, label="Cross-dataset F1\n(avg on other 2)",
                   color="#E53935", edgecolor="black")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=11, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=11, fontweight="bold")

    # Dessiner des flèches montrant la baisse
    for i in range(len(labels)):
        drop = same_ds_f1[i] - cross_ds_f1_avg[i]
        mid_x = x[i]
        ax.annotate(f"Drop: {drop:.0%}",
                    xy=(mid_x + width/2, cross_ds_f1_avg[i]),
                    xytext=(mid_x + 0.6, max(cross_ds_f1_avg[i] + 0.15, 0.55)),
                    fontsize=10, color="red", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xlabel("Training Dataset", fontsize=13)
    ax.set_ylabel("Macro-F1", fontsize=13)
    ax.set_title("Domain Shift: Same-Dataset vs Cross-Dataset Performance",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    out = Path(output_dir)
    plt.savefig(out / "domain_shift_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'domain_shift_summary.png'}")


def main():
    output_dir = "outputs/thesis_figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_cross_dataset_heatmap(output_dir)
    plot_leave_one_out(output_dir)
    plot_ablation_comparison(output_dir)
    plot_finetune_comparison(output_dir)
    plot_per_dataset_model_comparison(output_dir)
    plot_domain_shift_summary(output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
