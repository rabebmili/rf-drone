"""Audio Spectrogram Transformer (AST) pour la classification de spectrogrammes RF."""

import torch
import torch.nn as nn
import math


class ASTBlock(nn.Module):
    """Bloc encodeur Transformer standard."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class RFAST(nn.Module):
    """AST pour spectrogrammes RF monocanal avec tokenisation par patchs sans CNN."""

    def __init__(self, num_classes=2, patch_size=16, embed_dim=192,
                 num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Embedding par patchs : Conv2d avec kernel=stride=patch_size (projection linéaire)
        self.patch_embed = nn.Conv2d(
            1, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # Token CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Embedding positionnel (interpolé si la taille d'entrée change)
        self.num_patches_default = (257 // patch_size) * (511 // patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches_default + 1, embed_dim) * 0.02
        )

        # Encodeur Transformer
        self.blocks = nn.Sequential(
            *[ASTBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def _interpolate_pos_embed(self, num_patches):
        # Interpole les embeddings positionnels si la taille d'entrée diffère
        if num_patches == self.num_patches_default:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1, :]  # [1, 1, D]
        patch_pos = self.pos_embed[:, 1:, :]  # [1, N_default, D]

        # Interpolation 1D le long de la dimension séquentielle
        patch_pos = patch_pos.transpose(1, 2)  # [1, D, N_default]
        patch_pos = nn.functional.interpolate(
            patch_pos, size=num_patches, mode="linear", align_corners=False
        )
        patch_pos = patch_pos.transpose(1, 2)  # [1, num_patches, D]

        return torch.cat([cls_pos, patch_pos], dim=1)

    def _encode(self, x):
        # Encode l'entrée via embedding par patchs + transformer, retourne le token CLS
        B = x.shape[0]

        # Embedding par patchs : [B, 1, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.patch_embed(x)
        # Aplatir les dims spatiales en séquence : [B, embed_dim, N] -> [B, N, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        num_patches = x.shape[1]

        # Ajouter le token CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N+1, embed_dim]

        # Ajouter l'embedding positionnel
        pos_embed = self._interpolate_pos_embed(num_patches)
        x = x + pos_embed

        # Transformer
        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]  # Sortie du token CLS

    def forward(self, x):
        cls_out = self._encode(x)
        return self.head(cls_out)

    def get_embedding(self, x):
        # Retourne l'embedding du token CLS avant la tête de classification
        return self._encode(x)
