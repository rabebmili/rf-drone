"""Wrappers de datasets pour l'entraînement Siamese par triplets."""

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """Enveloppe un dataset de classification pour produire des triplets (ancre, positif, négatif)."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self._build_index()

    def _build_index(self):
        # Construit un index par classe pour un échantillonnage efficace
        self.class_indices = defaultdict(list)
        for i in range(len(self.base_dataset)):
            _, y = self.base_dataset[i]
            label = y.item() if isinstance(y, torch.Tensor) else y
            self.class_indices[label].append(i)
        self.classes = list(self.class_indices.keys())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        anchor_x, anchor_y = self.base_dataset[idx]
        anchor_label = anchor_y.item() if isinstance(anchor_y, torch.Tensor) else anchor_y

        # Positif : échantillon aléatoire de même classe
        pos_candidates = [i for i in self.class_indices[anchor_label] if i != idx]
        if not pos_candidates:
            pos_candidates = self.class_indices[anchor_label]
        pos_idx = random.choice(pos_candidates)
        positive_x, _ = self.base_dataset[pos_idx]

        # Négatif : échantillon aléatoire de classe différente
        neg_classes = [c for c in self.classes if c != anchor_label]
        neg_class = random.choice(neg_classes)
        neg_idx = random.choice(self.class_indices[neg_class])
        negative_x, _ = self.base_dataset[neg_idx]

        return anchor_x, positive_x, negative_x
