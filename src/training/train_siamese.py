"""Script d'entraînement du réseau Siamois avec triplet loss."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

from src.models import MODEL_REGISTRY, RAW_SIGNAL_MODELS
from src.models.siamese_network import SiameseNetwork
from src.datasets.siamese_dataset import TripletDataset
from src.training.train_multimodel import load_dataset


def plot_curves(history, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="tab:blue")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Triplet Loss")
    axes[0].set_title("Triplet Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_triplet_acc"], label="Val", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Triplet Accuracy")
    axes[1].set_title("Val Triplet Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0,
                    ema_state=None, ema_decay=0.0):
    model.train()
    running_loss = 0.0
    total = 0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if ema_state is not None and ema_decay > 0:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                    else:
                        ema_state[k].copy_(v.detach())

        running_loss += loss.item() * anchor.size(0)
        total += anchor.size(0)

    return running_loss / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)
            loss = criterion(anchor_emb, pos_emb, neg_emb)

            d_pos = (anchor_emb - pos_emb).pow(2).sum(dim=1)
            d_neg = (anchor_emb - neg_emb).pow(2).sum(dim=1)
            correct += (d_pos < d_neg).sum().item()

            running_loss += loss.item() * anchor.size(0)
            total += anchor.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Siamese network for drone attribution")
    parser.add_argument("--dataset", choices=["dronerf", "cagedronerf", "rfuav"],
                        default="dronerf")
    parser.add_argument("--backbone", default="resnet",
                        choices=[k for k in MODEL_REGISTRY if k not in RAW_SIGNAL_MODELS],
                        help="Backbone model for Siamese encoder (spectrogram-based only)")
    parser.add_argument("--task", choices=["binary", "multiclass"],
                        default="multiclass")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--backbone_weights", default=None,
                        help="Path to pretrained backbone weights")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear LR warmup epochs (clamped to epochs/4)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay on model weights (0 = disabled)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if args.output_dir is None:
        args.output_dir = f"outputs/siamese_{args.dataset}_{args.backbone}_{args.task}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chargement du jeu de données de base
    train_ds, val_ds, _, num_classes, class_names = load_dataset(
        args.dataset, args.task
    )

    # Encapsulation en dataset de triplets
    train_triplet = TripletDataset(train_ds)
    val_triplet = TripletDataset(val_ds)

    train_loader = DataLoader(train_triplet, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_triplet, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Modèle
    model = SiameseNetwork(
        backbone_name=args.backbone,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
    ).to(device)

    # Chargement du backbone pré-entraîné si fourni
    if args.backbone_weights:
        from src.models import get_model
        pretrained = get_model(args.backbone, num_classes=num_classes)
        pretrained.load_state_dict(
            torch.load(args.backbone_weights, weights_only=True, map_location=device)
        )
        model.backbone.load_state_dict(pretrained.state_dict())
        print(f"Loaded pretrained backbone from {args.backbone_weights}")

    criterion = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = min(args.warmup_epochs, args.epochs // 4)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs,
                                         eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    print(f"LR={args.lr:.2e}, warmup={warmup_epochs}, weight_decay={args.weight_decay}, "
          f"grad_clip={args.grad_clip}, ema_decay={args.ema_decay}")

    use_ema = args.ema_decay > 0
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if use_ema else None

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_triplet_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0

    print(f"Training Siamese ({args.backbone} backbone) on {args.dataset} ({args.task})")
    print(f"  Triplets: {len(train_triplet)} train, {len(val_triplet)} val")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=args.grad_clip,
            ema_state=ema_state, ema_decay=args.ema_decay,
        )

        if use_ema:
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            val_loss, val_triplet_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_triplet_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_triplet_acc"].append(val_triplet_acc)

        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Triplet Acc: {val_triplet_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_triplet_acc
            torch.save(model.state_dict(), model_dir / "best_siamese.pt")
            print(f"  -> Best model saved (val_loss={val_loss:.4f}, acc={val_triplet_acc:.4f})")

        if use_ema:
            model.load_state_dict(raw_state)

    plot_curves(history, str(figures_dir / "siamese_training_curves.png"))

    # Sauvegarde des résultats
    results = {
        "backbone": args.backbone,
        "dataset": args.dataset,
        "task": args.task,
        "embedding_dim": args.embedding_dim,
        "margin": args.margin,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "best_val_triplet_acc": best_val_acc,
        "num_classes": num_classes,
        "class_names": class_names,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
