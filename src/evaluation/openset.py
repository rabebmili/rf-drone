"""Open-set recognition and OOD detection (MSP, Energy, Mahalanobis)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pathlib import Path


def compute_msp_scores(model, loader, device):
    """Maximum Softmax Probability scores. Lower = more likely OOD."""
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_probs = probs.max(dim=1).values
            scores.extend(max_probs.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(scores), np.array(labels)


def compute_energy_scores(model, loader, device, temperature=1.0):
    """Energy-based OOD score. Higher = more in-distribution (sign-flipped for MSP consistency)."""
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            energy = temperature * torch.logsumexp(logits / temperature, dim=1)
            scores.extend(energy.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(scores), np.array(labels)


def compute_mahalanobis_scores(model, loader, device, class_means, shared_cov_inv):
    """Mahalanobis distance in embedding space. Higher = more in-distribution (sign-flipped)."""
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            embeddings = model.get_embedding(x).cpu().numpy()

            for emb in embeddings:
                # Distance de Mahalanobis minimale parmi les classes connues
                min_dist = float("inf")
                for mean in class_means:
                    diff = emb - mean
                    dist = diff @ shared_cov_inv @ diff
                    min_dist = min(min_dist, dist)
                scores.append(-min_dist)  # inverser : plus élevé = plus en distribution

            labels.extend(y.numpy())

    return np.array(scores), np.array(labels)


def fit_mahalanobis(model, train_loader, device, num_classes):
    """Compute per-class means and shared covariance from training embeddings."""
    model.eval()
    embeddings_by_class = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            embs = model.get_embedding(x).cpu().numpy()
            for emb, label in zip(embs, y.numpy()):
                embeddings_by_class[label].append(emb)

    class_means = []
    all_centered = []

    for c in range(num_classes):
        embs = np.array(embeddings_by_class[c])
        mean = embs.mean(axis=0)
        class_means.append(mean)
        all_centered.append(embs - mean)

    all_centered = np.concatenate(all_centered, axis=0)
    shared_cov = np.cov(all_centered, rowvar=False) + 1e-5 * np.eye(all_centered.shape[1])
    shared_cov_inv = np.linalg.inv(shared_cov)

    return class_means, shared_cov_inv


def evaluate_ood_detection(in_scores, ood_scores, method_name="MSP"):
    """Compute OOD detection metrics: AUROC, AUPR, FPR@95TPR."""
    labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    # TFP à 95% TVP
    sorted_in = np.sort(in_scores)
    threshold = sorted_in[int(0.05 * len(sorted_in))]  # 5e percentile de la distribution connue
    fpr_95 = np.mean(ood_scores >= threshold)

    print(f"  {method_name:>15} | AUROC: {auroc:.4f} | AUPR: {aupr:.4f} | FPR@95: {fpr_95:.4f}")

    return {"auroc": auroc, "aupr": aupr, "fpr_at_95tpr": fpr_95}


def create_openset_split(dataset, holdout_class):
    """Split dataset into known (in-distribution) and unknown (OOD) subsets."""
    known_idx = []
    unknown_idx = []

    for i in range(len(dataset)):
        _, label = dataset[i]
        if label.item() == holdout_class:
            unknown_idx.append(i)
        else:
            known_idx.append(i)

    return known_idx, unknown_idx


def run_openset_evaluation(model, test_dataset, device, holdout_class,
                            train_loader=None, num_known_classes=None,
                            output_dir=None):
    """Run full open-set evaluation with MSP, Energy, and optionally Mahalanobis."""
    known_idx, unknown_idx = create_openset_split(test_dataset, holdout_class)

    known_ds = Subset(test_dataset, known_idx)
    unknown_ds = Subset(test_dataset, unknown_idx)

    known_loader = DataLoader(known_ds, batch_size=16, shuffle=False, num_workers=0)
    unknown_loader = DataLoader(unknown_ds, batch_size=16, shuffle=False, num_workers=0)

    results = {}

    # PSM (Probabilité Softmax Maximale)
    in_msp, _ = compute_msp_scores(model, known_loader, device)
    ood_msp, _ = compute_msp_scores(model, unknown_loader, device)
    results["MSP"] = evaluate_ood_detection(in_msp, ood_msp, "MSP")

    # Énergie
    in_energy, _ = compute_energy_scores(model, known_loader, device)
    ood_energy, _ = compute_energy_scores(model, unknown_loader, device)
    results["Energy"] = evaluate_ood_detection(in_energy, ood_energy, "Energy")

    # Mahalanobis (nécessite la méthode get_embedding)
    if hasattr(model, "get_embedding") and train_loader is not None and num_known_classes is not None:
        class_means, cov_inv = fit_mahalanobis(model, train_loader, device, num_known_classes)
        in_maha, _ = compute_mahalanobis_scores(model, known_loader, device, class_means, cov_inv)
        ood_maha, _ = compute_mahalanobis_scores(model, unknown_loader, device, class_means, cov_inv)
        results["Mahalanobis"] = evaluate_ood_detection(in_maha, ood_maha, "Mahalanobis")

    # Tracer les distributions de scores
    if output_dir:
        _plot_ood_distributions(in_msp, ood_msp, "MSP Score", output_dir, "msp_distribution.png")
        _plot_ood_distributions(in_energy, ood_energy, "Energy Score", output_dir, "energy_distribution.png")

    return results


def _plot_ood_distributions(in_scores, ood_scores, score_name, output_dir, filename):
    """Plot in-distribution vs OOD score histograms."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(in_scores, bins=50, alpha=0.6, label="In-distribution (known)", density=True)
    ax.hist(ood_scores, bins=50, alpha=0.6, label="OOD (unknown)", density=True)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.set_title(f"OOD Detection — {score_name}")
    ax.legend()
    plt.tight_layout()

    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
