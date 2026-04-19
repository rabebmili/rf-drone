"""Dataset de graphes de signaux pour l'entraînement GNN (noeuds = embeddings, arêtes = similarité cosinus)."""

import torch
from torch.utils.data import Dataset

from src.models.gnn import build_similarity_graph


class SignalGraphDataset(Dataset):
    """Échantillonne des sous-graphes aléatoires depuis l'ensemble d'embeddings.

    En mode `deterministic=True`, les sous-graphes sont pré-échantillonnés une seule
    fois (seed fixée) et mis en cache, afin que les métriques de validation/test
    soient reproductibles d'une époque à l'autre (sinon chaque appel tire de
    nouveaux noeuds et les courbes oscillent artificiellement).
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        subgraph_size: int = 64,
        threshold: float = 0.5,
        k: int = 5,
        num_samples: int = 500,
        deterministic: bool = False,
        seed: int = 0,
    ):
        self.embeddings = embeddings.float()
        self.labels = labels.long()
        self.subgraph_size = min(subgraph_size, len(embeddings))
        self.threshold = threshold
        self.k = k
        self.num_samples = num_samples
        self.deterministic = deterministic

        if deterministic:
            gen = torch.Generator().manual_seed(seed)
            N = len(self.embeddings)
            self._cache = []
            for _ in range(num_samples):
                perm = torch.randperm(N, generator=gen)[:self.subgraph_size]
                sub_emb = self.embeddings[perm]
                sub_lbl = self.labels[perm]
                adj = build_similarity_graph(
                    sub_emb, threshold=self.threshold, k=self.k, add_self_loops=True
                )
                self._cache.append((sub_emb, adj, sub_lbl))
        else:
            self._cache = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]

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
