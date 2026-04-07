"""Réseau siamois pour l'attribution de drones par similarité."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import get_model


class SiameseNetwork(nn.Module):
    """Réseau siamois enveloppant un backbone pour l'attribution par similarité, entraîné avec triplet loss."""

    def __init__(self, backbone_name="resnet", num_classes=4,
                 embedding_dim=128, freeze_backbone=False):
        super().__init__()
        self.backbone_name = backbone_name

        # Charger le backbone et obtenir sa dimension d'embedding
        self.backbone = get_model(backbone_name, num_classes=num_classes)

        # Déterminer la dimension d'embedding via une passe avant fictive
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1, 64, 64)
            emb = self.backbone.get_embedding(dummy)
            backbone_dim = emb.shape[1]

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Tête de projection : projette les embeddings dans un espace normalisé
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def encode(self, x):
        # Encode une entrée vers son embedding projeté
        emb = self.backbone.get_embedding(x)
        proj = self.projection(emb)
        # Normalisation L2 pour la similarité cosinus
        return F.normalize(proj, p=2, dim=1)

    def forward(self, x):
        # Passe avant standard : retourne l'embedding normalisé L2
        return self.encode(x)

    def forward_triplet(self, anchor, positive, negative):
        # Passe avant pour l'entraînement triplet, retourne (ancre, positif, négatif)
        return self.encode(anchor), self.encode(positive), self.encode(negative)

    def get_embedding(self, x):
        # Retourne l'embedding projeté et normalisé
        return self.encode(x)

    def compute_similarity(self, x1, x2):
        # Calcule la similarité cosinus entre deux entrées
        emb1 = self.encode(x1)
        emb2 = self.encode(x2)
        return F.cosine_similarity(emb1, emb2)
