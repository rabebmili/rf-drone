"""Construction d'une galerie d'embeddings de drones connus pour l'attribution Siamese."""

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import MODEL_REGISTRY, RAW_SIGNAL_MODELS
from src.models.siamese_network import SiameseNetwork
from src.training.train_multimodel import load_dataset


def build_gallery(model, dataset, device, class_names):
    # Calcule les embeddings moyens par classe à partir d'un encodeur Siamese
    model.eval()
    embeddings_by_class = defaultdict(list)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            embs = model.get_embedding(x).cpu().numpy()
            for emb, label in zip(embs, y.numpy()):
                embeddings_by_class[int(label)].append(emb)

    # Calculer l'embedding moyen par classe
    mean_embeddings = []
    used_class_names = []
    for cls_idx in sorted(embeddings_by_class.keys()):
        embs = np.array(embeddings_by_class[cls_idx])
        mean_emb = embs.mean(axis=0)
        # Normalisation L2
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        mean_embeddings.append(mean_emb)
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        used_class_names.append(name)

    return np.array(mean_embeddings), used_class_names


def main():
    parser = argparse.ArgumentParser(description="Build known-drone embedding gallery")
    parser.add_argument("--dataset", choices=["dronerf", "cagedronerf", "rfuav"],
                        default="dronerf")
    parser.add_argument("--task", choices=["binary", "multiclass"],
                        default="multiclass")
    parser.add_argument("--siamese_weights", required=True,
                        help="Path to trained Siamese model weights")
    parser.add_argument("--backbone", default="resnet",
                        choices=[k for k in MODEL_REGISTRY if k not in RAW_SIGNAL_MODELS],
                        help="Backbone model used in Siamese network (spectrogram-based only)")
    parser.add_argument("--output", default=None,
                        help="Output .npz file path")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"outputs/gallery_{args.dataset}_{args.task}.npz"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger le jeu de données (données d'entraînement pour la galerie)
    train_ds, _, _, num_classes, class_names = load_dataset(args.dataset, args.task)

    # Charger le modèle Siamese
    model = SiameseNetwork(
        backbone_name=args.backbone,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(
        torch.load(args.siamese_weights, weights_only=True, map_location=device)
    )
    model.eval()

    # Construire la galerie
    print(f"Building gallery from {args.dataset} ({args.task}) training data...")
    embeddings, gallery_names = build_gallery(model, train_ds, device, class_names)

    # Sauvegarder
    np.savez(
        args.output,
        embeddings=embeddings,
        class_names=np.array(gallery_names),
    )
    print(f"Gallery saved: {args.output}")
    print(f"  Classes: {gallery_names}")
    print(f"  Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
