"""Hybride CNN-Transformer pour la classification de spectrogrammes."""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class RFTransformer(nn.Module):
    """Hybride CNN-Transformer pour la classification de spectrogrammes RF monocanal."""

    def __init__(self, num_classes=2, embed_dim=128, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Tronc CNN pour réduire les dimensions spatiales
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.token_pool = nn.AdaptiveAvgPool2d((8, 16))  # -> 128 tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 8 * 16 + 1, embed_dim) * 0.02)

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, dropout=dropout)
              for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def _encode(self, x):
        # Passe avant partagée (tronc + transformer), retourne l'embedding CLS
        B = x.shape[0]

        x = self.stem(x)
        x = self.token_pool(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 129, embed_dim]
        x = x + self.pos_embed

        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        cls_out = self._encode(x)
        return self.head(cls_out)

    def get_embedding(self, x):
        # Retourne l'embedding du token CLS avant la tête de classification
        return self._encode(x)
