"""Script d'entraînement du réseau Siamois avec triplet loss."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.models.siamese_network import SiameseNetwork
from src.datasets.siamese_dataset import TripletDataset
from src.training.train_multimodel import load_dataset


def train_one_epoch(model, loader, criterion, optimizer, device):
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
        optimizer.step()

        running_loss += loss.item() * anchor.size(0)
        total += anchor.size(0)

    return running_loss / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)
            loss = criterion(anchor_emb, pos_emb, neg_emb)

            running_loss += loss.item() * anchor.size(0)
            total += anchor.size(0)

    return running_loss / total


def main():
    parser = argparse.ArgumentParser(description="Train Siamese network for drone attribution")
    parser.add_argument("--dataset", choices=["dronerf", "cagedronerf", "rfuav"],
                        default="dronerf")
    parser.add_argument("--backbone", default="resnet",
                        help="Backbone model for Siamese encoder")
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
    args = parser.parse_args()

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(f"Training Siamese ({args.backbone} backbone) on {args.dataset} ({args.task})")
    print(f"  Triplets: {len(train_triplet)} train, {len(val_triplet)} val")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / "best_siamese.pt")
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")

    # Sauvegarde des résultats
    results = {
        "backbone": args.backbone,
        "dataset": args.dataset,
        "task": args.task,
        "embedding_dim": args.embedding_dim,
        "margin": args.margin,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "num_classes": num_classes,
        "class_names": class_names,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
