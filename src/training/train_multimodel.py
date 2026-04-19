"""Script d'entraînement unifié pour tous les modèles, jeux de données et tâches."""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.models import MODEL_REGISTRY, RAW_SIGNAL_MODELS, get_model
from src.evaluation.metrics import full_evaluation, collect_predictions

# Modèles nécessitant un LR plus faible : attention + CNN 1D (signaux bruts très sensibles)
ATTENTION_MODELS = {"ast", "conformer", "transformer", "cnn1d"}


def compute_class_weights(dataset):
    # Calcule les poids de classe par fréquence inverse
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.item() if isinstance(y, torch.Tensor) else y)
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(counts)
    weights = torch.zeros(max(counts.keys()) + 1)
    for cls_id, count in counts.items():
        weights[cls_id] = total / (num_classes * count)
    return weights


def load_dataset(dataset_name, task, data_path=None, model_name=None):
    # Charge les splits train/val/test pour le jeu de données spécifié
    if dataset_name == "dronerf":
        # Dataset brut pour les modèles 1D
        if model_name and model_name in RAW_SIGNAL_MODELS:
            from src.datasets.dronerf_raw_dataset import DroneRFRawDataset

            csv_path = data_path or "data/metadata/dronerf_segments_split.csv"

            if task == "binary":
                label_col = "label_binary"
                num_classes = 2
                class_names = ["Background", "Drone"]
            else:
                label_col = "label_multiclass"
                num_classes = 4
                class_names = ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]

            train_ds = DroneRFRawDataset(csv_path, split="train", label_col=label_col)
            val_ds = DroneRFRawDataset(csv_path, split="val", label_col=label_col)
            test_ds = DroneRFRawDataset(csv_path, split="test", label_col=label_col)

            return train_ds, val_ds, test_ds, num_classes, class_names

        from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset

        csv_path = data_path or "data/metadata/dronerf_precomputed.csv"

        if task == "binary":
            label_col = "label_binary"
            num_classes = 2
            class_names = ["Background", "Drone"]
        else:
            label_col = "label_multiclass"
            num_classes = 4
            class_names = ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]

        train_ds = DroneRFPrecomputedDataset(csv_path, split="train", label_col=label_col)
        val_ds = DroneRFPrecomputedDataset(csv_path, split="val", label_col=label_col)
        test_ds = DroneRFPrecomputedDataset(csv_path, split="test", label_col=label_col)

    elif dataset_name == "cagedronerf":
        from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders

        root = data_path or "data/raw/CageDroneRF/balanced"
        train_ds, val_ds, test_ds = create_cagedronerf_loaders(
            root, label_mode=task, augment_train=True
        )

        if task == "binary":
            num_classes = 2
            class_names = ["Background/Non-drone", "Drone"]
        else:
            num_classes = train_ds.num_classes
            class_names = train_ds.get_class_names()

    elif dataset_name == "rfuav":
        from src.datasets.rfuav_dataset import RFUAVDataset, create_rfuav_splits

        root = data_path or "data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train"

        if task == "binary":
            train_ds, val_ds = create_rfuav_splits(root, val_ratio=0.2, label_mode="binary")
            test_ds = val_ds
            num_classes = 2
            class_names = ["Background", "Drone"]

        else:
            train_ds, val_ds = create_rfuav_splits(root, val_ratio=0.2, label_mode="multiclass")
            test_ds = val_ds
            num_classes = train_ds.num_classes
            class_names = train_ds.get_class_names()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds, test_ds, num_classes, class_names


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0,
                    ema_state=None, ema_decay=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Écrêtage du gradient pour éviter les explosions de gradient
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # EMA : mise à jour par-batch (plus efficace que par-epoch)
        if ema_state is not None and ema_decay > 0:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                    else:
                        ema_state[k].copy_(v.detach())

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return running_loss / total, correct / total


def _ema_smooth(values, alpha=0.3):
    """Moyenne mobile exponentielle pour lisser les courbes de validation."""
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


def plot_curves(history, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Courbes de perte (train + val)
    axes[0].plot(epochs, history["train_loss"], label="Train", color="tab:blue")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Courbes d'accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", color="tab:blue")
    axes[1].plot(epochs, history["val_acc"], label="Val", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    # Val Macro-F1 seul (raw)
    axes[2].plot(epochs, history["val_f1"], label="Val Macro-F1", color="tab:orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Val Macro-F1")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RF drone classifier")
    parser.add_argument("--dataset", choices=["dronerf", "cagedronerf", "rfuav"],
                        default="dronerf", help="Dataset to train on")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        default="smallrf", help="Model architecture")
    parser.add_argument("--task", choices=["binary", "multiclass"],
                        default="binary", help="Classification task")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 3e-4 for attention models, 1e-3 for CNN)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of linear LR warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=0.5,
                        help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing for CrossEntropyLoss (default: 0.0 for binary, 0.05 for multiclass)")
    parser.add_argument("--data_path", default=None, help="Override default data path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_weighted_loss", action="store_true",
                        help="Disable class-weighted loss")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay for weight averaging (0 = disabled, try 0.999)")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # LR par défaut selon le type de modèle : plus faible pour les modèles à attention
    if args.lr is None:
        args.lr = 3e-4 if args.model in ATTENTION_MODELS else 5e-4

    # Pas de lissage de label pour la classification binaire (tâche simple)
    if args.label_smoothing is None:
        args.label_smoothing = 0.0 if args.task == "binary" else 0.05

    if args.output_dir is None:
        prefix = f"{args.dataset}_"
        args.output_dir = f"outputs/{prefix}{args.model}_{args.task}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chargement des données
    train_ds, val_ds, test_ds, num_classes, class_names = load_dataset(
        args.dataset, args.task, args.data_path, model_name=args.model
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Modèle
    model = get_model(args.model, num_classes=num_classes).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Perte pondérée par classe si déséquilibre + lissage de label pour régularisation
    # val_criterion : sans pondération ni lissage → perte de validation comparable entre modèles
    val_criterion = nn.CrossEntropyLoss()
    if not args.no_weighted_loss:
        class_weights = compute_class_weights(train_ds).to(device)
        weight_ratio = class_weights.max() / class_weights.min()
        if weight_ratio > 1.5:
            criterion = nn.CrossEntropyLoss(weight=class_weights,
                                            label_smoothing=args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # AdamW avec weight decay pour une meilleure régularisation
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Warmup linéaire suivi d'un cosine annealing pour stabiliser l'entraînement
    warmup_epochs = min(args.warmup_epochs, args.epochs // 4)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.lr * 0.01)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    print(f"LR={args.lr:.2e}, warmup={warmup_epochs} epochs, "
          f"weight_decay={args.weight_decay}, grad_clip={args.grad_clip}, "
          f"label_smoothing={args.label_smoothing}, ema_decay={args.ema_decay}")

    # EMA : shadow weights mises à jour chaque epoch, utilisées pour la validation
    use_ema = args.ema_decay > 0
    if use_ema:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Boucle d'entraînement
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    best_val_f1 = 0.0  # suivi pour le reporting final uniquement
    out_dir = Path(args.output_dir)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                grad_clip=args.grad_clip,
                                                ema_state=ema_state if use_ema else None,
                                                ema_decay=args.ema_decay)

        if use_ema:
            # Validation sur les poids EMA (courbes lisses)
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            val_loss, val_acc = validate(model, val_loader, val_criterion, device)
            y_true, y_pred, _ = collect_predictions(model, val_loader, device, return_probs=False)
            val_f1 = f1_score(y_true, y_pred, average="macro")
        else:
            val_loss, val_acc = validate(model, val_loader, val_criterion, device)
            y_true, y_pred, _ = collect_predictions(model, val_loader, device, return_probs=False)
            val_f1 = f1_score(y_true, y_pred, average="macro")

        scheduler.step()
        best_val_f1 = max(best_val_f1, val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Sauvegarde du meilleur modèle selon la val loss (signal plus fiable que le F1)
        # Avec EMA, on sauvegarde les poids EMA (déjà chargés dans le modèle ci-dessus)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_dir = out_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            best_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved (Val Loss={val_loss:.4f})")

        # Restaurer les poids d'entraînement pour continuer
        if use_ema:
            model.load_state_dict(raw_state)

    # Chargement du meilleur modèle pour évaluation finale
    model.load_state_dict(torch.load(out_dir / "models" / "best_model.pt", weights_only=True))

    figures_dir = out_dir / "figures"
    metrics, y_true, y_pred, y_prob = full_evaluation(
        model, test_loader, device,
        class_names=class_names,
        output_dir=str(figures_dir),
        model_name=f"{args.model} {args.dataset} ({args.task})"
    )

    # Sauvegarde des courbes d'entraînement
    plot_curves(history, str(figures_dir / "training_curves.png"))

    # Sauvegarde des résultats finaux
    results_path = out_dir / "results.json"
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report"] = metrics["classification_report"]
    serializable["model"] = args.model
    serializable["dataset"] = args.dataset
    serializable["task"] = args.task
    serializable["epochs"] = args.epochs
    serializable["best_val_loss"] = best_val_loss
    serializable["best_val_f1"] = best_val_f1
    serializable["param_count"] = param_count

    # Conversion des types numpy
    for k, v in serializable.items():
        if hasattr(v, "item"):
            serializable[k] = v.item()

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
