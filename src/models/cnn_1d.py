"""CNN 1D pour le traitement direct de signaux RF I/Q bruts (sans STFT)."""

import torch
import torch.nn as nn


class RFCNN1D(nn.Module):
    """CNN 1D pour la classification de signaux RF bruts sans conversion en spectrogramme."""

    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1 : sous-échantillonnage agressif pour les longues séquences
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            # Bloc 2
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            # Bloc 3
            nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            # Bloc 4
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            # Bloc 5
            nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _encode(self, x):
        # Extrait les caractéristiques du signal brut
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze(-1)  # [B, 128]
        return x

    def forward(self, x):
        features = self._encode(x)
        return self.classifier(features)

    def get_embedding(self, x):
        # Retourne le vecteur de 128 dimensions avant la tête de classification
        return self._encode(x)
