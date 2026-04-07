"""Ensemble CNN + Transformer pour améliorer la robustesse."""

import torch
import torch.nn as nn

from src.models import get_model


class EnsembleCNNTransformer(nn.Module):
    """Ensemble fusionnant un CNN et un Transformer avec trois stratégies : average, weighted, stacking."""

    def __init__(self, num_classes=2, cnn_name="resnet", transformer_name="ast",
                 fusion="average"):
        super().__init__()
        self.fusion = fusion
        self.num_classes = num_classes

        self.cnn = get_model(cnn_name, num_classes=num_classes)
        self.transformer = get_model(transformer_name, num_classes=num_classes)

        if fusion == "weighted":
            self.alpha = nn.Parameter(torch.tensor(0.5))

        elif fusion == "stacking":
            # Obtenir les dimensions d'embedding via une passe avant fictive
            self.cnn.eval()
            self.transformer.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 1, 64, 64)
                cnn_dim = self.cnn.get_embedding(dummy).shape[1]
                tf_dim = self.transformer.get_embedding(dummy).shape[1]

            self.stacking_head = nn.Sequential(
                nn.Linear(cnn_dim + tf_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
            self._cnn_dim = cnn_dim
            self._tf_dim = tf_dim

    def forward(self, x):
        if self.fusion == "stacking":
            cnn_emb = self.cnn.get_embedding(x)
            tf_emb = self.transformer.get_embedding(x)
            combined = torch.cat([cnn_emb, tf_emb], dim=1)
            return self.stacking_head(combined)

        cnn_logits = self.cnn(x)
        tf_logits = self.transformer(x)

        if self.fusion == "average":
            return (cnn_logits + tf_logits) / 2.0
        elif self.fusion == "weighted":
            alpha = torch.sigmoid(self.alpha)
            return alpha * cnn_logits + (1 - alpha) * tf_logits
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

    def get_embedding(self, x):
        # Retourne les embeddings concaténés des deux sous-modèles
        cnn_emb = self.cnn.get_embedding(x)
        tf_emb = self.transformer.get_embedding(x)
        return torch.cat([cnn_emb, tf_emb], dim=1)

    def load_pretrained(self, cnn_weights_path, transformer_weights_path, device="cpu"):
        # Charge les poids pré-entraînés des deux sous-modèles
        self.cnn.load_state_dict(
            torch.load(cnn_weights_path, weights_only=True, map_location=device)
        )
        self.transformer.load_state_dict(
            torch.load(transformer_weights_path, weights_only=True, map_location=device)
        )
