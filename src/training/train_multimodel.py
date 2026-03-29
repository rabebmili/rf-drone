"""Unified training script for all models x datasets x tasks."""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.models.cnn_spectrogram import SmallRFNet
from src.models.resnet_spectrogram import RFResNet
from src.models.transformer_spectrogram import RFTransformer
from src.evaluation.metrics import full_evaluation, collect_predictions


MODEL_REGISTRY = {
    "smallrf": SmallRFNet,
    "resnet": RFResNet,
    "transformer": RFTransformer,
}


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights from a dataset."""
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


def load_dataset(dataset_name, task, data_path=None):
    """Load train/val/test splits for the specified dataset."""
    if dataset_name == "dronerf":
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


def train_one_epoch(model, loader, criterion, optimizer, device):
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
        optimizer.step()

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


def plot_curves(history, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["val_f1"], label="Val Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Validation Macro-F1")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_path", default=None, help="Override default data path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_weighted_loss", action="store_true",
                        help="Disable class-weighted loss")
    args = parser.parse_args()

    if args.output_dir is None:
        prefix = f"{args.dataset}_" if args.dataset != "dronerf" else ""
        args.output_dir = f"outputs/{prefix}{args.model}_{args.task}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_ds, val_ds, test_ds, num_classes, class_names = load_dataset(
        args.dataset, args.task, args.data_path
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(num_classes=num_classes).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Class-weighted loss if imbalanced
    if not args.no_weighted_loss:
        class_weights = compute_class_weights(train_ds).to(device)
        weight_ratio = class_weights.max() / class_weights.min()
        if weight_ratio > 1.5:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    out_dir = Path(args.output_dir)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Validation F1
        y_true, y_pred, _ = collect_predictions(model, val_loader, device, return_probs=False)
        val_f1 = f1_score(y_true, y_pred, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_dir = out_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            best_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved (F1={val_f1:.4f})")

    # Load best model for final test evaluation
    model.load_state_dict(torch.load(out_dir / "models" / "best_model.pt", weights_only=True))

    figures_dir = out_dir / "figures"
    metrics, y_true, y_pred, y_prob = full_evaluation(
        model, test_loader, device,
        class_names=class_names,
        output_dir=str(figures_dir),
        model_name=f"{args.model} {args.dataset} ({args.task})"
    )

    # Save training curves
    plot_curves(history, str(figures_dir / "training_curves.png"))

    # Save final results
    results_path = out_dir / "results.json"
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report"] = metrics["classification_report"]
    serializable["model"] = args.model
    serializable["dataset"] = args.dataset
    serializable["task"] = args.task
    serializable["epochs"] = args.epochs
    serializable["best_val_f1"] = best_val_f1
    serializable["param_count"] = param_count

    # Convert numpy types
    for k, v in serializable.items():
        if hasattr(v, "item"):
            serializable[k] = v.item()

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
