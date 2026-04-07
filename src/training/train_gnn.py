"""Entraînement du GNN pour l'investigation par graphe de similarité RF."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.models.gnn import RFDroneGNN
from src.datasets.signal_graph_dataset import SignalGraphDataset, collate_graphs
from src.training.train_multimodel import load_dataset


def extract_embeddings(backbone, loader, device):
    # Extrait les embeddings normalisés L2 et les labels depuis un loader
    backbone.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]
            x = x.to(device)
            emb = backbone.get_embedding(x)
            emb = nn.functional.normalize(emb, dim=-1)
            all_emb.append(emb.cpu())
            all_lbl.append(y)
    return torch.cat(all_emb), torch.cat(all_lbl)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for emb, adj, lbl in loader:
        emb, adj, lbl = emb.to(device), adj.to(device), lbl.to(device)
        B, N, D = emb.shape

        # Traitement de chaque graphe du batch
        batch_loss = 0.0
        for i in range(B):
            logits = model(emb[i], adj[i])  # [N, C]
            batch_loss = batch_loss + criterion(logits, lbl[i])

        batch_loss = batch_loss / B
        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for emb, adj, lbl in loader:
        emb, adj, lbl = emb.to(device), adj.to(device), lbl.to(device)
        B, N, D = emb.shape
        for i in range(B):
            logits = model(emb[i], adj[i])
            preds = logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl[i].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    return acc, f1, mcc


def plot_curves(train_losses, val_accs, val_f1s, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(val_accs, label='Val Accuracy')
    ax2.plot(val_f1s, label='Val Macro-F1')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / 'gnn_training_curves.png', dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dronerf', choices=['dronerf', 'cagedronerf', 'rfuav'])
    parser.add_argument('--task', default='multiclass', choices=['binary', 'multiclass'])
    parser.add_argument('--siamese_weights', type=str, default=None,
                        help='Path to trained Siamese model weights')
    parser.add_argument('--backbone', default='resnet',
                        help='Backbone model name (used if no siamese_weights)')
    parser.add_argument('--backbone_weights', type=str, default=None,
                        help='Path to backbone classifier weights')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--subgraph_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=5,
                        help='k-NN minimum connectivity per node')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_samples', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Répertoire de sortie ────────────────────────────────────────────────
    out_tag = f"gnn_{args.dataset}_{args.task}"
    out_dir = Path(f"outputs/{out_tag}")
    (out_dir / 'models').mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(parents=True, exist_ok=True)

    # ── Chargement du jeu de données ────────────────────────────────────────
    print("Chargement du jeu de données...")
    train_ds, val_ds, test_ds, num_classes, class_names = load_dataset(
        args.dataset, args.task
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

    # ── Chargement du backbone pour extraction d'embeddings ────────────────
    if args.siamese_weights and Path(args.siamese_weights).exists():
        print(f"Loading Siamese model from {args.siamese_weights}")
        from src.models.siamese_network import SiameseNetwork
        siamese = SiameseNetwork(backbone_name=args.backbone, num_classes=num_classes)
        siamese.load_state_dict(torch.load(args.siamese_weights, map_location=device))
        backbone = siamese.backbone
        emb_dim = siamese.embedding_dim
    else:
        print(f"Loading backbone classifier: {args.backbone}")
        from src.models import get_model
        backbone = get_model(args.backbone, num_classes)
        if args.backbone_weights and Path(args.backbone_weights).exists():
            backbone.load_state_dict(torch.load(args.backbone_weights, map_location=device))
            print(f"  Loaded weights from {args.backbone_weights}")
        # Détermination de la dimension d'embedding
        dummy = torch.zeros(1, 1, 128, 128)
        with torch.no_grad():
            emb_dim = backbone.get_embedding(dummy).shape[-1]

    backbone = backbone.to(device)

    # ── Extraction des embeddings ──────────────────────────────────────────
    print("Extraction des embeddings...")
    train_emb, train_lbl = extract_embeddings(backbone, train_loader, device)
    val_emb,   val_lbl   = extract_embeddings(backbone, val_loader,   device)
    test_emb,  test_lbl  = extract_embeddings(backbone, test_loader,  device)
    emb_dim = train_emb.shape[-1]
    print(f"  Embedding dim: {emb_dim}, Train: {len(train_emb)}, Val: {len(val_emb)}, Test: {len(test_emb)}")

    # ── Construction des datasets de graphes ───────────────────────────────
    train_graph_ds = SignalGraphDataset(
        train_emb, train_lbl,
        subgraph_size=args.subgraph_size,
        threshold=args.threshold,
        k=args.k,
        num_samples=args.num_samples,
    )
    val_graph_ds = SignalGraphDataset(
        val_emb, val_lbl,
        subgraph_size=args.subgraph_size,
        threshold=args.threshold,
        k=args.k,
        num_samples=100,
    )
    test_graph_ds = SignalGraphDataset(
        test_emb, test_lbl,
        subgraph_size=args.subgraph_size,
        threshold=args.threshold,
        k=args.k,
        num_samples=200,
    )

    gnn_train = DataLoader(train_graph_ds, batch_size=4, shuffle=True,
                           num_workers=0, collate_fn=collate_graphs)
    gnn_val   = DataLoader(val_graph_ds,   batch_size=4, shuffle=False,
                           num_workers=0, collate_fn=collate_graphs)
    gnn_test  = DataLoader(test_graph_ds,  batch_size=4, shuffle=False,
                           num_workers=0, collate_fn=collate_graphs)

    # ── Modèle GNN ────────────────────────────────────────────────────────────
    model = RFDroneGNN(
        in_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_heads=args.num_heads,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GNN parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Boucle d'entraînement ──────────────────────────────────────────────────
    best_f1, best_epoch = 0.0, 0
    train_losses, val_accs, val_f1s = [], [], []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, gnn_train, optimizer, criterion, device)
        acc, f1, mcc = eval_epoch(model, gnn_val, device)
        scheduler.step()

        train_losses.append(loss)
        val_accs.append(acc)
        val_f1s.append(f1)

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss {loss:.4f} | Val Acc {acc:.4f} | Val F1 {f1:.4f} | MCC {mcc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / 'models' / 'best_gnn.pt')

    # ── Évaluation sur le test ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(out_dir / 'models' / 'best_gnn.pt', map_location=device))
    test_acc, test_f1, test_mcc = eval_epoch(model, gnn_test, device)
    print(f"\nTest: Accuracy: {test_acc:.4f} | Macro-F1: {test_f1:.4f} | MCC: {test_mcc:.4f}")

    # ── Sauvegarde des résultats ────────────────────────────────────────────────
    results = {
        'accuracy': test_acc,
        'macro_f1': test_f1,
        'mcc': test_mcc,
        'best_val_f1': best_f1,
        'best_epoch': best_epoch,
        'model': 'gnn',
        'backbone': args.backbone,
        'dataset': args.dataset,
        'task': args.task,
        'epochs': args.epochs,
        'param_count': total_params,
        'emb_dim': emb_dim,
        'subgraph_size': args.subgraph_size,
        'threshold': args.threshold,
        'k': args.k,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plot_curves(train_losses, val_accs, val_f1s, out_dir / 'figures')
    print(f"\nResults saved to {out_dir}/")


if __name__ == '__main__':
    main()
