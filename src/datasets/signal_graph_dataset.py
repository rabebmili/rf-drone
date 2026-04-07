"""Dataset de graphes de signaux pour l'entraînement GNN (noeuds = embeddings, arêtes = similarité cosinus)."""

import torch
from torch.utils.data import Dataset

from src.models.gnn import build_similarity_graph


class SignalGraphDataset(Dataset):
    """Échantillonne des sous-graphes aléatoires depuis l'ensemble d'embeddings."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        subgraph_size: int = 64,
        threshold: float = 0.5,
        k: int = 5,
        num_samples: int = 500,
    ):
        self.embeddings = embeddings.float()
        self.labels = labels.long()
        self.subgraph_size = min(subgraph_size, len(embeddings))
        self.threshold = threshold
        self.k = k
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        N = len(self.embeddings)
        size = self.subgraph_size

        # Échantillonner un sous-ensemble aléatoire de noeuds
        perm = torch.randperm(N)[:size]
        sub_emb = self.embeddings[perm]    # [size, D]
        sub_lbl = self.labels[perm]        # [size]

        adj = build_similarity_graph(
            sub_emb, threshold=self.threshold, k=self.k, add_self_loops=True
        )

        return sub_emb, adj, sub_lbl


def collate_graphs(batch):
    # Collate une liste de (emb, adj, labels) en tenseurs empilés
    embs, adjs, labels = zip(*batch)
    return torch.stack(embs), torch.stack(adjs), torch.stack(labels)
