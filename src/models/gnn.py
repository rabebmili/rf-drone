"""Réseau de neurones sur graphes pour l'investigation RF de drones."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Couche Graph Attention : calcule h_i' = sigma( sum_j alpha_ij * W * h_j )."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Passe avant : x [N, in_dim], adj [N, N] -> [N, out_dim]
        N = x.size(0)
        h = self.W(x)  # [N, out_dim]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, H, D]

        # Scores d'attention
        e_src = (h * self.attn_src).sum(-1)  # [N, H]
        e_dst = (h * self.attn_dst).sum(-1)  # [N, H]

        # e[i,j] = LeakyReLU(a_src[i] + a_dst[j])
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)  # [N, N, H]
        e = self.leaky_relu(e)

        # Masquer les non-arêtes
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, float('-inf'))

        alpha = torch.softmax(e, dim=1)  # [N, N, H]
        alpha = self.dropout(alpha)

        # Agrégation
        out = torch.einsum('ijh,jhd->ihd', alpha, h)  # [N, H, D]
        return out.reshape(N, -1)  # [N, out_dim]


class RFDroneGNN(nn.Module):
    """GNN pour l'investigation de signaux RF de drones, propage l'information entre segments liés."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.gat1 = GATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Passe avant : x [N, in_dim], adj [N, N] -> logits [N, num_classes]
        h = F.relu(self.in_proj(x))
        h = self.dropout(h)

        # Bloc GAT 1
        h = self.norm1(h + F.elu(self.gat1(h, adj)))

        # Bloc GAT 2
        h = self.norm2(h + F.elu(self.gat2(h, adj)))

        return self.classifier(h)

    def get_embedding(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Retourne les embeddings raffinés par le graphe (avant la tête de classification)
        h = F.relu(self.in_proj(x))
        h = self.dropout(h)
        h = self.norm1(h + F.elu(self.gat1(h, adj)))
        h = self.norm2(h + F.elu(self.gat2(h, adj)))
        return h


def build_similarity_graph(
    embeddings: torch.Tensor,
    threshold: float = 0.5,
    k: int = 5,
    add_self_loops: bool = True,
) -> torch.Tensor:
    # Construit la matrice d'adjacence par similarité cosinus entre embeddings
    # Similarité cosinus (les embeddings doivent être normalisés L2)
    emb = F.normalize(embeddings, dim=-1)
    sim = emb @ emb.T  # [N, N]

    # Arêtes par seuil
    adj = (sim >= threshold).float()

    # Repli k-NN : garantir au moins k voisins par noeud
    if k > 0:
        topk_vals, topk_idx = sim.topk(min(k + 1, sim.size(1)), dim=1)
        knn_adj = torch.zeros_like(adj)
        knn_adj.scatter_(1, topk_idx, 1.0)
        adj = torch.clamp(adj + knn_adj, max=1.0)

    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        adj = torch.clamp(adj, max=1.0)

    return adj
