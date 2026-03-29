"""Robustness evaluation under varying SNR levels with Gaussian noise injection."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path

from src.evaluation.metrics import collect_predictions, compute_classification_metrics


class NoisyDatasetWrapper(Dataset):
    """Wraps a spectrogram dataset and adds Gaussian noise at a given SNR."""

    def __init__(self, base_dataset, snr_db):
        self.base = base_dataset
        self.snr_db = snr_db

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]

        if self.snr_db is not None:
            # x est un tenseur [1, H, W]
            signal_power = torch.mean(x ** 2)
            snr_linear = 10 ** (self.snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = torch.randn_like(x) * torch.sqrt(noise_power)
            x = x + noise

        return x, y


def evaluate_robustness(model, test_dataset, device, snr_levels,
                        batch_size=16, class_names=None):
    """Evaluate model at each SNR level, return {snr_key: metrics_dict}."""
    results = {}

    # Évaluation sans bruit
    clean_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, y_prob = collect_predictions(model, clean_loader, device)
    clean_metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    results["clean"] = clean_metrics
    # Évaluations avec bruit
    for snr in snr_levels:
        noisy_ds = NoisyDatasetWrapper(test_dataset, snr_db=snr)
        noisy_loader = DataLoader(noisy_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        y_true, y_pred, y_prob = collect_predictions(model, noisy_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
        results[f"snr_{snr}dB"] = metrics

    return results


def plot_robustness_curves(results, snr_levels, output_path=None, model_name="Model"):
    """Plot accuracy and F1 vs SNR."""
    accs = [results["clean"]["accuracy"]]
    f1s = [results["clean"]["macro_f1"]]
    x_labels = ["Clean"]

    for snr in snr_levels:
        key = f"snr_{snr}dB"
        accs.append(results[key]["accuracy"])
        f1s.append(results[key]["macro_f1"])
        x_labels.append(f"{snr} dB")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(x_labels))
    ax.plot(x, accs, "o-", label="Accuracy", linewidth=2, markersize=6)
    ax.plot(x, f1s, "s-", label="Macro F1", linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("SNR Level")
    ax.set_ylabel("Score")
    ax.set_title(f"Robustness vs. SNR — {model_name}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.close()


def run_robustness_evaluation(model, test_dataset, device, output_dir,
                               model_name="Model", class_names=None,
                               snr_levels=None):
    """Full robustness evaluation with plots."""
    if snr_levels is None:
        snr_levels = [30, 20, 10, 5, 0, -5, -10]

    print(f"\n{'='*60}")
    print(f"  EVALUATION DE ROBUSTESSE : {model_name}")
    print(f"{'='*60}")

    results = evaluate_robustness(model, test_dataset, device, snr_levels,
                                   class_names=class_names)

    out_dir = Path(output_dir)
    plot_robustness_curves(results, snr_levels,
                           output_path=str(out_dir / "robustness_vs_snr.png"),
                           model_name=model_name)

    return results
