"""Détection open-set et OOD (MSP, Energy, Mahalanobis, OpenMax)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import exponweib
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path


def compute_msp_scores(model, loader, device):
    # Scores de probabilité softmax maximale. Plus bas = plus probablement OOD
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
    # Score OOD basé sur l'énergie. Plus élevé = plus en distribution (signe inversé)
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
    # Distance de Mahalanobis dans l'espace d'embeddings. Plus élevé = plus en distribution
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
    # Calculer les moyennes par classe et la covariance partagée depuis les embeddings d'entraînement
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


def fit_openmax(model, train_loader, device, num_classes, tail_size=20):
    # Ajuster OpenMax : calculer les MAV par classe + modèles Weibull via EVT
    model.eval()
    activations_by_class = {c: [] for c in range(num_classes)}
    logits_by_class = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            embeddings = model.get_embedding(x).cpu().numpy()
            logits_np = logits.cpu().numpy()

            for emb, logit, pred, label in zip(embeddings, logits_np, preds.cpu().numpy(), y.numpy()):
                # Utiliser uniquement les échantillons correctement classifiés
                if pred == label:
                    activations_by_class[label].append(emb)
                    logits_by_class[label].append(logit)

    # Calculer les MAV et ajuster les distributions de Weibull
    mavs = []
    weibull_params = []

    for c in range(num_classes):
        if not activations_by_class[c]:
            mavs.append(np.zeros(1))
            weibull_params.append((1.0, 0.0, 1.0))
            continue

        embs = np.array(activations_by_class[c])
        mav = embs.mean(axis=0)
        mavs.append(mav)

        # Calculer les distances de chaque échantillon à son MAV de classe
        distances = np.linalg.norm(embs - mav, axis=1)

        # Utiliser uniquement la queue (plus grandes distances) pour l'ajustement Weibull
        sorted_distances = np.sort(distances)
        tail = sorted_distances[-min(tail_size, len(sorted_distances)):]

        # Ajuster la distribution de Weibull aux distances de queue
        try:
            if len(tail) < 3:
                raise ValueError(f"Trop peu d'échantillons de queue ({len(tail)}) pour l'ajustement Weibull")
            # Normaliser les distances pour éviter les problèmes numériques
            tail_max = tail.max() if tail.max() > 0 else 1.0
            tail_norm = tail / tail_max
            params = exponweib.fit(tail_norm, floc=0)
            # Re-scaler le paramètre d'échelle
            params = (params[0], params[1], params[2], params[3] * tail_max)
            weibull_params.append(params)
        except Exception:
            # Fallback : paramètres Weibull par défaut basés sur la distribution empirique
            mean_d = tail.mean() if len(tail) > 0 else 1.0
            weibull_params.append((1.0, 1.0, 0.0, mean_d))

    return mavs, weibull_params


def compute_openmax_scores(model, loader, device, mavs, weibull_params,
                           num_classes=None, alpha=3):
    # Calculer les probabilités OpenMax incluant la classe 'inconnu'
    if num_classes is None:
        num_classes = len(mavs)

    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            embeddings = model.get_embedding(x).cpu().numpy()

            for emb, logit in zip(embeddings, logits):
                # Distances à chaque MAV de classe
                distances = np.array([
                    np.linalg.norm(emb - mavs[c]) for c in range(num_classes)
                ])

                # Trier les classes par distance (plus proche en premier)
                ranked_classes = np.argsort(distances)

                # Réviser les activations des top-alpha classes via CDF de Weibull
                revised_logits = logit.copy()
                unknown_score = 0.0

                for rank, c in enumerate(ranked_classes[:alpha]):
                    dist = distances[c]
                    try:
                        w_cdf = exponweib.cdf(dist, *weibull_params[c])
                    except Exception:
                        w_cdf = 0.0

                    # Réduire l'activation de la classe connue, accumuler pour l'inconnu
                    reduction = logit[c] * w_cdf
                    revised_logits[c] -= reduction
                    unknown_score += reduction

                # Ajouter le score inconnu comme classe supplémentaire
                revised_with_unknown = np.append(revised_logits, unknown_score)

                # Softmax
                exp_logits = np.exp(revised_with_unknown - revised_with_unknown.max())
                openmax_probs = exp_logits / exp_logits.sum()

                # Score : 1 - P(inconnu). Plus élevé = plus en distribution
                scores.append(1.0 - openmax_probs[-1])

            labels.extend(y.numpy())

    return np.array(scores), np.array(labels)


def evaluate_ood_detection(in_scores, ood_scores, method_name="MSP"):
    # Calculer les métriques de détection OOD : AUROC, AUPR, TFP@95TVP
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
    # Diviser le dataset en sous-ensembles connus (en distribution) et inconnus (OOD)
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
    # Évaluation open-set complète avec MSP, Energy, et optionnellement Mahalanobis
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

    # OpenMax (reconnaissance open-set basée sur EVT)
    if hasattr(model, "get_embedding") and train_loader is not None and num_known_classes is not None:
        try:
            mavs, weibull_params = fit_openmax(model, train_loader, device, num_known_classes)
            in_openmax, _ = compute_openmax_scores(
                model, known_loader, device, mavs, weibull_params, num_known_classes
            )
            ood_openmax, _ = compute_openmax_scores(
                model, unknown_loader, device, mavs, weibull_params, num_known_classes
            )
            results["OpenMax"] = evaluate_ood_detection(in_openmax, ood_openmax, "OpenMax")

            if output_dir:
                _plot_ood_distributions(
                    in_openmax, ood_openmax, "OpenMax Score",
                    output_dir, "openmax_distribution.png"
                )
        except Exception as e:
            import traceback
            print(f"  OpenMax échoué : {e}")
            traceback.print_exc()

    # Tracer les distributions de scores
    if output_dir:
        _plot_ood_distributions(in_msp, ood_msp, "MSP Score", output_dir, "msp_distribution.png")
        _plot_ood_distributions(in_energy, ood_energy, "Energy Score", output_dir, "energy_distribution.png")
        if "Mahalanobis" in results:
            _plot_ood_distributions(in_maha, ood_maha, "Mahalanobis Score", output_dir, "mahalanobis_distribution.png")

    return results


def _plot_ood_distributions(in_scores, ood_scores, score_name, output_dir, filename):
    # Tracer les histogrammes des scores en distribution vs OOD
    import numpy as np
    def _safe_hist(ax, scores, **kwargs):
        """Plot histogram; fall back to fewer bins if range is too small."""
        bins = kwargs.pop("bins", 50)
        for b in [bins, 20, 10, 5, 1]:
            try:
                ax.hist(scores, bins=b, **kwargs)
                return
            except ValueError:
                continue

    fig, ax = plt.subplots(figsize=(8, 4))
    _safe_hist(ax, in_scores,  bins=50, alpha=0.6, label="In-distribution (known)", density=True)
    _safe_hist(ax, ood_scores, bins=50, alpha=0.6, label="OOD (unknown)", density=True)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.set_title(f"OOD Detection — {score_name}")
    ax.legend()
    plt.tight_layout()

    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
