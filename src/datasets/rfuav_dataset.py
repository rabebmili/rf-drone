"""RFUAV dataset loader for pre-generated spectrogram JPG images (37 drone models)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class RFUAVDataset(Dataset):
    """Loads RFUAV JPG spectrograms as single-channel normalized tensors."""

    def __init__(self, root_dir, target_size=(257, 511),
                 label_mode="binary", indices=None):
        """Initialize dataset from a directory of drone class subfolders."""
        self.root = Path(root_dir)
        self.target_size = target_size
        self.label_mode = label_mode

        if not self.root.exists():
            raise FileNotFoundError(
                f"RFUAV directory not found: {self.root}\n"
                f"Download: python -m src.datasets.download_rfuav"
            )

        # Découvrir les classes de drones (sous-dossiers)
        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes) if label_mode == "multiclass" else 2

        # Collecter tous les chemins d'images + étiquettes (JPG ou PNG)
        self._all_samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            # Chercher les images dans le dossier de classe ou le sous-dossier imgs/
            img_dir = cls_dir / "imgs" if (cls_dir / "imgs").exists() else cls_dir
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in sorted(img_dir.glob(ext)):
                    if label_mode == "binary":
                        label = 1  # tous les échantillons RFUAV sont des drones
                    else:
                        label = self.class_to_idx[cls_name]
                    self._all_samples.append((str(img_path), label))

        # Appliquer le filtre d'indices (pour les splits entraînement/val)
        if indices is not None:
            self.samples = [self._all_samples[i] for i in indices]
        else:
            self.samples = self._all_samples

        # Preload all images into RAM
        self._cache = []
        for img_path, label in self.samples:
            img = Image.open(img_path).convert("L")
            if self.target_size is not None:
                img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - arr.mean()) / (arr.std() + 1e-10)
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
            y = torch.tensor(label, dtype=torch.long)
            self._cache.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._cache[idx]

    def get_class_names(self):
        return self.classes.copy()

    def get_total_samples(self):
        return len(self._all_samples)


def create_rfuav_splits(root_dir, val_ratio=0.2, random_state=42, **kwargs):
    """Create stratified train/val split from a single RFUAV folder."""

    # Charger le dataset complet pour obtenir le nombre d'échantillons et les étiquettes
    full = RFUAVDataset(root_dir, **kwargs)
    n = full.get_total_samples()
    all_labels = [full._all_samples[i][1] for i in range(n)]

    # Split stratifié par indices (préserve les proportions de classes)
    train_idx, val_idx = train_test_split(
        list(range(n)),
        test_size=val_ratio,
        random_state=random_state,
        stratify=all_labels
    )

    # Créer les datasets filtrés à partir des indices du split
    train_ds = RFUAVDataset(root_dir, indices=train_idx, **kwargs)
    val_ds = RFUAVDataset(root_dir, indices=val_idx, **kwargs)

    return train_ds, val_ds
