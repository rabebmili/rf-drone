"""Conformer pour la classification de spectrogrammes RF."""

import torch
import torch.nn as nn


class FeedForwardModule(nn.Module):
    """Module feed-forward avec expansion et dropout."""

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    """Module de convolution Conformer : pointwise -> depthwise -> pointwise."""

    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            # Transposer pour Conv1d : [B, T, D] -> [B, D, T]
            Rearrange(),
            nn.Conv1d(dim, 2 * dim, kernel_size=1),  # expansion pointwise
            nn.GLU(dim=1),  # retour à dim canaux
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim),  # depthwise
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),  # projection pointwise
            nn.Dropout(dropout),
            # Retransposer : [B, D, T] -> [B, T, D]
            RearrangeBack(),
        )

    def forward(self, x):
        return self.net(x)


class Rearrange(nn.Module):
    """[B, T, D] -> [B, D, T]"""
    def forward(self, x):
        return x.transpose(1, 2)


class RearrangeBack(nn.Module):
    """[B, D, T] -> [B, T, D]"""
    def forward(self, x):
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Bloc Conformer : FFN -> MHSA -> Conv -> FFN (avec résidus demi-pas)."""

    def __init__(self, dim, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, dropout=dropout)
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, dropout=dropout)
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, x):
        # Feed-forward demi-pas
        x = x + 0.5 * self.ff1(x)

        # Auto-attention multi-têtes
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Module de convolution
        x = x + self.conv(x)

        # Feed-forward demi-pas
        x = x + 0.5 * self.ff2(x)

        # Normalisation finale
        x = self.norm_final(x)
        return x


class RFConformer(nn.Module):
    """Conformer pour la classification de spectrogrammes RF monocanal."""

    def __init__(self, num_classes=2, embed_dim=128, num_heads=4,
                 num_layers=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Frontend CNN pour réduire les dimensions spatiales
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Pooling adaptatif vers taille fixe avant aplatissement
        self.spatial_pool = nn.AdaptiveAvgPool2d((8, 16))  # -> 128 tokens

        # Projeter les caractéristiques concaténées vers embed_dim
        self.proj = nn.Linear(64, embed_dim)

        # Blocs Conformer
        self.blocks = nn.Sequential(
            *[ConformerBlock(embed_dim, num_heads, conv_kernel_size, dropout)
              for _ in range(num_layers)]
        )

        # Tête de classification (pooling sur la séquence puis classification)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def _encode(self, x):
        # Encode l'entrée en un vecteur de caractéristiques
        B = x.shape[0]

        # Frontend CNN : [B, 1, H, W] -> [B, 64, H/4, W/4]
        x = self.frontend(x)

        # Pooling vers taille spatiale fixe : [B, 64, 8, 16]
        x = self.spatial_pool(x)

        # Aplatir en séquence : [B, 64, 8, 16] -> [B, 128, 64]
        x = x.flatten(2).transpose(1, 2)  # [B, 128, 64]

        # Projeter vers embed_dim : [B, 128, embed_dim]
        x = self.proj(x)

        # Blocs Conformer
        x = self.blocks(x)

        # Pooling moyen global sur la séquence
        x = x.mean(dim=1)  # [B, embed_dim]

        return x

    def forward(self, x):
        features = self._encode(x)
        return self.classifier(features)

    def get_embedding(self, x):
        # Retourne le vecteur de caractéristiques avant la tête de classification
        return self._encode(x)
