"""Auto-encodeur variationnel pour la détection d'anomalies sur spectrogrammes RF."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFVAE(nn.Module):
    """VAE convolutif pour spectrogrammes RF monocanal, détection d'anomalies par erreur de reconstruction."""

    def __init__(self, num_classes=None, latent_dim=32):
        super().__init__()
        # num_classes non utilisé (non supervisé), conservé pour compatibilité d'interface
        self.latent_dim = latent_dim

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> H/2, W/2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> H/4, W/4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> H/8, W/8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> H/16, W/16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Pooling adaptatif vers taille fixe avant les couches FC
        self.enc_pool = nn.AdaptiveAvgPool2d((4, 4))
        enc_flat_dim = 128 * 4 * 4  # 2048

        self.fc_mu = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, latent_dim)

        # Décodeur
        self.fc_decode = nn.Linear(latent_dim, enc_flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        # Encode l'entrée vers les paramètres de la distribution latente
        h = self.encoder(x)
        h = self.enc_pool(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Astuce de reparamétrisation : z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, target_size=None):
        # Décode le vecteur latent en reconstruction
        h = self.fc_decode(z)
        h = h.view(-1, 128, 4, 4)
        h = self.decoder(h)
        # Redimensionner pour correspondre aux dimensions d'entrée
        if target_size is not None:
            h = F.interpolate(h, size=target_size, mode="bilinear", align_corners=False)
        return h

    def forward(self, x):
        # Passe avant complète : encoder -> reparamétriser -> décoder, retourne (x_recon, mu, logvar)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, target_size=x.shape[2:])
        return x_recon, mu, logvar

    def get_embedding(self, x):
        # Retourne le vecteur latent moyen (mu)
        mu, _ = self.encode(x)
        return mu

    def anomaly_score(self, x):
        # Calcule l'erreur de reconstruction par échantillon (MSE), score élevé = anomalie probable
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            # MSE par échantillon
            mse = F.mse_loss(x_recon, x, reduction="none")
            scores = mse.view(mse.size(0), -1).mean(dim=1)
        return scores

    @staticmethod
    def loss_function(x_recon, x, mu, logvar, beta=0.5):
        # Perte VAE : reconstruction (MSE) + beta * divergence KL
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
