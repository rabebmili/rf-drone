"""EfficientNet-B0 adapté pour la classification de spectrogrammes RF monocanal."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class RFEfficientNet(nn.Module):
    """EfficientNet-B0 avec entrée monocanal pour la classification de spectrogrammes."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Adapter la première conv de 3 à 1 canal
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialiser la conv 1 canal avec la moyenne des poids pré-entraînés 3 canaux
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        # Remplacer la tête de classification
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        self._embedding_dim = in_features

    def _encode(self, x):
        # Extrait les caractéristiques avant le classifieur
        features = self.backbone(x)
        return features

    def forward(self, x):
        features = self._encode(x)
        return self.classifier(features)

    def get_embedding(self, x):
        # Retourne le vecteur de 1280 dimensions avant la tête de classification
        return self._encode(x)
