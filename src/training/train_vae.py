"""Script d'entraînement du VAE pour la détection d'anomalies."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.vae import RFVAE
from src.training.train_multimodel import load_dataset


def train_one_epoch(model, loader, optimizer, device, beta=0.5):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total = 0

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = RFVAE.loss_function(x_recon, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_recon += recon_loss.item() * x.size(0)
        total_kl += kl_loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / total, total_recon / total, total_kl / total


def validate(model, loader, device, beta=0.5):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = RFVAE.loss_function(x_recon, x, mu, logvar, beta=beta)

            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)
            total += x.size(0)

    return total_loss / total, total_recon / total, total_kl / total


def plot_vae_curves(history, output_path):
    # Trace les courbes d'entraînement du VAE : perte totale, reconstruction, KL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss (Recon + KL)")
    axes[0].legend()

    axes[1].plot(epochs, history["train_recon"], label="Train")
    axes[1].plot(epochs, history["val_recon"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].set_title("Reconstruction Loss (MSE)")
    axes[1].legend()

    axes[2].plot(epochs, history["train_kl"], label="Train")
    axes[2].plot(epochs, history["val_kl"], label="Val")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("KL Divergence")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_reconstructions(model, dataset, device, output_path, n_samples=5):
    # Affiche les spectrogrammes originaux vs reconstruits
    model.eval()
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6))

    for i in range(n_samples):
        x, _ = dataset[i]
        x_input = x.unsqueeze(0).to(device)

        with torch.no_grad():
            x_recon, _, _ = model(x_input)

        orig = x.squeeze(0).cpu().numpy()
        recon = x_recon.squeeze(0).squeeze(0).cpu().numpy()

        axes[0, i].imshow(orig, aspect="auto", origin="lower", cmap="viridis")
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon, aspect="auto", origin="lower", cmap="viridis")
        axes[1, i].set_title(f"Reconstructed {i}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.suptitle("VAE Reconstructions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train VAE for anomaly detection")
    parser.add_argument("--dataset", choices=["dronerf", "cagedronerf", "rfuav"],
                        default="dronerf")
    parser.add_argument("--task", choices=["binary", "multiclass"],
                        default="multiclass")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.5,
                        help="KL weight (beta-VAE)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/vae_{args.dataset}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chargement du jeu de données
    train_ds, val_ds, test_ds, num_classes, class_names = load_dataset(
        args.dataset, args.task
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Modèle
    model = RFVAE(latent_dim=args.latent_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE parameters: {param_count:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl": [], "val_kl": [],
    }
    best_val_loss = float("inf")

    print(f"Training VAE on {args.dataset} (latent_dim={args.latent_dim}, beta={args.beta})")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, beta=args.beta
        )
        val_loss, val_recon, val_kl = validate(
            model, val_loader, device, beta=args.beta
        )
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)
        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)

        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"Train: {train_loss:.4f} (R={train_recon:.4f} KL={train_kl:.4f}) | "
              f"Val: {val_loss:.4f} (R={val_recon:.4f} KL={val_kl:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / "best_vae.pt")
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")

    # Courbes d'entraînement
    plot_vae_curves(history, str(figures_dir / "vae_training_curves.png"))

    # Visualisation des reconstructions
    model.load_state_dict(torch.load(model_dir / "best_vae.pt", weights_only=True))
    plot_reconstructions(model, val_ds, device, str(figures_dir / "vae_reconstructions.png"))

    # Sauvegarde des résultats
    results = {
        "dataset": args.dataset,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "param_count": param_count,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
