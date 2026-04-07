"""Métriques de classification, courbes ROC/PR, matrice de confusion et calibration (ECE)."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score
)


CLASS_NAMES_BINARY = ["Background", "Drone"]
CLASS_NAMES_MULTI = ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]


def collect_predictions(model, loader, device, return_probs=True):
    # Exécuter le modèle sur un DataLoader, retourner prédictions, étiquettes et probabilités
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs) if return_probs else None

    return y_true, y_pred, y_prob


def compute_classification_metrics(y_true, y_pred, y_prob=None, class_names=None):
    # Calculer les métriques de classification complètes
    # Utiliser toutes les étiquettes présentes dans y_true et y_pred
    all_labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    num_classes = len(all_labels)
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=all_labels, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, labels=all_labels,
            target_names=class_names[:num_classes], zero_division=0
        ),
    }

    if y_prob is not None and num_classes == 2:
        results["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
    elif y_prob is not None and num_classes > 2:
        try:
            results["roc_auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            results["roc_auc_ovr"] = None

    if y_prob is not None:
        results["ece"] = compute_ece(y_true, y_pred, y_prob)

    return results


def compute_ece(y_true, y_pred, y_prob, n_bins=15):
    # Erreur de calibration attendue (ECE)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return ece


def plot_confusion_matrix(y_true, y_pred, class_names=None, output_path=None,
                          title="Confusion Matrix", normalize=None):
    # Tracer et sauvegarder la matrice de confusion, adaptée au nombre de classes
    unique_labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    num_classes = len(unique_labels)
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    display_labels = [class_names[i] if i < len(class_names) else str(i)
                      for i in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    if normalize == "true":
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    # Adapter la taille de la figure en fonction du nombre de classes
    if num_classes <= 6:
        figsize = (7, 6)
        fontsize = 12
    elif num_classes <= 15:
        figsize = (12, 10)
        fontsize = 9
    else:
        figsize = (18, 15)
        fontsize = 7

    fig, ax = plt.subplots(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", values_format=fmt,
              text_kw={"fontsize": max(5, fontsize - 2)})
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(display_labels, fontsize=fontsize)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names=None, output_path=None):
    # Tracer les courbes ROC (binaire ou un-contre-reste), résumé si >10 classes
    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    fig, ax = plt.subplots(figsize=(10, 7))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"Drone (AUC = {roc_auc:.4f})")
    elif num_classes <= 10:
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.4f})")
    else:
        # Beaucoup de classes : calculer l'AUC par classe, afficher macro-moyenne + meilleur/pire
        class_aucs = {}
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() == 0:
                continue
            fpr_i, tpr_i, _ = roc_curve(y_binary, y_prob[:, i])
            class_aucs[i] = (fpr_i, tpr_i, auc(fpr_i, tpr_i))

        # ROC macro-moyenne
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        for i, (fpr_i, tpr_i, _) in class_aucs.items():
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= len(class_aucs)
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, linewidth=3, color="navy",
                label=f"Macro-average (AUC = {macro_auc:.3f})")

        # Afficher les 5 pires classes (les plus informatives)
        sorted_by_auc = sorted(class_aucs.items(), key=lambda x: x[1][2])
        for i, (fpr_i, tpr_i, auc_i) in sorted_by_auc[:5]:
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            ax.plot(fpr_i, tpr_i, alpha=0.6, linewidth=1,
                    label=f"{label} (AUC = {auc_i:.3f})")

        # Statistiques résumées
        all_aucs = [v[2] for v in class_aucs.values()]
        ax.set_title(f"ROC Curves — {num_classes} classes\n"
                     f"AUC: mean={np.mean(all_aucs):.4f}, "
                     f"min={np.min(all_aucs):.4f}, max={np.max(all_aucs):.4f}",
                     fontsize=11)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if num_classes <= 10:
        ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def plot_precision_recall_curves(y_true, y_prob, class_names=None, output_path=None):
    # Tracer les courbes précision-rappel, résumé si >10 classes
    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    fig, ax = plt.subplots(figsize=(10, 7))

    if num_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])
        ax.plot(rec, prec, linewidth=2, label=f"Drone (AP = {ap:.4f})")
    elif num_classes <= 10:
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_binary, y_prob[:, i])
            ap = average_precision_score(y_binary, y_prob[:, i])
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            ax.plot(rec, prec, label=f"{label} (AP = {ap:.4f})")
    else:
        # Beaucoup de classes : afficher macro-moyenne + les 5 pires
        class_aps = {}
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() == 0:
                continue
            prec_i, rec_i, _ = precision_recall_curve(y_binary, y_prob[:, i])
            ap_i = average_precision_score(y_binary, y_prob[:, i])
            class_aps[i] = (prec_i, rec_i, ap_i)

        # Macro-moyenne
        all_aps = [v[2] for v in class_aps.values()]
        macro_ap = np.mean(all_aps)

        # Afficher les 5 pires (les plus informatives)
        sorted_by_ap = sorted(class_aps.items(), key=lambda x: x[1][2])
        for i, (prec_i, rec_i, ap_i) in sorted_by_ap[:5]:
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            ax.plot(rec_i, prec_i, alpha=0.7, linewidth=1.5,
                    label=f"{label} (AP = {ap_i:.3f})")

        ax.set_title(f"Precision-Recall Curves — {num_classes} classes\n"
                     f"mAP: {macro_ap:.4f}, "
                     f"min={np.min(all_aps):.4f}, max={np.max(all_aps):.4f}",
                     fontsize=11)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if num_classes <= 10:
        ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def plot_calibration_diagram(y_true, y_prob, n_bins=10, output_path=None):
    # Tracer le diagramme de fiabilité pour l'analyse de calibration
    confidences = np.max(y_prob, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black", label="Model")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title("Calibration Diagram")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.close()


def print_metrics_summary(metrics, model_name="Model"):
    # Afficher le résumé formaté des métriques de classification
    print(f"\n{'='*60}")
    print(f"  RÉSULTATS D'ÉVALUATION : {model_name}")
    print(f"{'='*60}")
    print(f"  Exactitude :            {metrics['accuracy']:.4f}")
    print(f"  Exactitude équilibrée : {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Macro :              {metrics['macro_f1']:.4f}")
    print(f"  F1 Pondéré :            {metrics['weighted_f1']:.4f}")
    print(f"  Precision Macro :       {metrics['macro_precision']:.4f}")
    print(f"  Rappel Macro :          {metrics['macro_recall']:.4f}")
    print(f"  Kappa de Cohen :        {metrics['cohen_kappa']:.4f}")
    print(f"  MCC :                   {metrics['mcc']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    if "roc_auc_ovr" in metrics and metrics["roc_auc_ovr"] is not None:
        print(f"  ROC-AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
    if "ece" in metrics:
        print(f"  ECE:                {metrics['ece']:.4f}")
    print(f"{'='*60}")
    print(f"\n{metrics['classification_report']}")


def full_evaluation(model, loader, device, class_names=None, output_dir=None,
                    model_name="Model"):
    # Évaluation complète : métriques + tous les graphiques
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    print_metrics_summary(metrics, model_name)

    if output_dir:
        out = Path(output_dir)
        plot_confusion_matrix(y_true, y_pred, class_names,
                              output_path=out / "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, class_names,
                              output_path=out / "confusion_matrix_normalized.png",
                              title="Confusion Matrix (Normalized)",
                              normalize="true")
        plot_roc_curves(y_true, y_prob, class_names,
                        output_path=out / "roc_curves.png")
        plot_precision_recall_curves(y_true, y_prob, class_names,
                                     output_path=out / "pr_curves.png")
        plot_calibration_diagram(y_true, y_prob,
                                 output_path=out / "calibration_diagram.png")

    return metrics, y_true, y_pred, y_prob
