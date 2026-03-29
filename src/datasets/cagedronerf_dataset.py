"""CageDroneRF dataset loader for pre-generated spectrogram PNG images."""

from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


# Les 27 classes du jeu de données équilibré (triées alphabétiquement)
CAGEDRONERF_CLASSES = [
    "Autel_EXOII",
    "Autel_Xstar",
    "DJI_FPV",
    "DJI_Inspire1",
    "DJI_Inspire2",
    "DJI_Mavic2Pro",
    "DJI_Mavic3",
    "DJI_MavicAir",
    "DJI_MavicMini4",
    "DJI_MavicMini4-armed",
    "DJI_MavicPro",
    "DJI_Mini3",
    "DJI_Mini3-armed",
    "DJI_Phantom3Adv",
    "DJI_Phantom4",
    "DJI_Tello",
    "HolyStone_HS110G",
    "HolyStone_HS720E",
    "Hubsan_X4_Air",
    "Laptop_Wi-Fi_video",
    "Parrot_Anafi",
    "Quadcopter_KY601S",
    "RadioMaster_TX16S",
    "Ruko_F11GIM",
    "Skydio_2",
    "Yuneec_Q500-HD",
    "background",
]

# Classes qui ne sont PAS des drones (pour l'étiquetage binaire)
NON_DRONE_CLASSES = {"background", "Laptop_Wi-Fi_video", "RadioMaster_TX16S"}


class CageDroneRFDataset(Dataset):
    """Loads CageDroneRF PNG spectrograms as single-channel normalized tensors."""

    def __init__(self, root_dir, target_size=(257, 511),
                 label_mode="binary", indices=None, augment=False):
        """Initialize dataset from a directory of class subfolders containing spectrogram images."""
        self.root = Path(root_dir)
        self.target_size = target_size
        self.label_mode = label_mode
        self.augment = augment

        if not self.root.exists():
            raise FileNotFoundError(
                f"CageDroneRF directory not found: {self.root}\n"
                f"Download from: https://drive.google.com/drive/folders/1B1QC3vAZqKB61TEXaqJsUhwgynbbphAp"
            )

        # Découvrir les classes (sous-dossiers)
        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes) if label_mode == "multiclass" else 2

        # Collecter tous les chemins d'images + étiquettes
        self._all_samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_path in sorted(cls_dir.glob(ext)):
                    if label_mode == "binary":
                        label = 0 if cls_name in NON_DRONE_CLASSES else 1
                    else:
                        label = self.class_to_idx[cls_name]
                    self._all_samples.append((str(img_path), label))

        # Appliquer le filtre d'indices
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
        x, y = self._cache[idx]
        if self.augment:
            x = self._apply_augmentation(x.clone())
        return x, y

    def _apply_augmentation(self, x):
        """Apply SpecAugment-style augmentation (time/frequency masking + noise)."""
        _, H, W = x.shape

        # Masquage fréquentiel (masquer des bandes horizontales aléatoires)
        if torch.rand(1).item() < 0.5:
            f = int(torch.randint(5, max(6, H // 8), (1,)).item())
            f0 = int(torch.randint(0, max(1, H - f), (1,)).item())
            x[:, f0:f0+f, :] = 0.0

        # Masquage temporel (masquer des bandes verticales aléatoires)
        if torch.rand(1).item() < 0.5:
            t = int(torch.randint(10, max(11, W // 6), (1,)).item())
            t0 = int(torch.randint(0, max(1, W - t), (1,)).item())
            x[:, :, t0:t0+t] = 0.0

        # Bruit gaussien
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(x) * 0.1
            x = x + noise

        return x

    def get_class_names(self):
        return self.classes.copy()

    def get_total_samples(self):
        return len(self._all_samples)

    def get_binary_class_names(self):
        return ["Background/Non-drone", "Drone"]

    def get_multiclass_class_names(self):
        return self.classes.copy()

    def get_class_weights(self):
        """Compute inverse-frequency class weights for weighted loss."""
        labels = [self._cache[i][1].item() for i in range(len(self._cache))]
        counts = Counter(labels)
        total = len(labels)
        num_classes = self.num_classes
        weights = torch.zeros(num_classes)
        for cls_id, count in counts.items():
            weights[cls_id] = total / (num_classes * count)
        return weights


def create_cagedronerf_loaders(root_dir="data/raw/CageDroneRF/balanced",
                                label_mode="binary", batch_size=16,
                                target_size=(257, 511), augment_train=False,
                                test_ratio=0.5):
    """Create train/val/test datasets by splitting the CageDroneRF balanced set."""
    root = Path(root_dir)
    train_images = root / "train" / "images"
    val_images = root / "val" / "images"

    # Jeu d'entraînement (avec augmentation optionnelle)
    train_ds = CageDroneRFDataset(
        train_images, target_size=target_size, label_mode=label_mode,
        augment=augment_train
    )

    # Charger le jeu de validation complet, puis diviser en val + test
    full_val = CageDroneRFDataset(
        val_images, target_size=target_size, label_mode=label_mode
    )

    n = full_val.get_total_samples()
    all_labels = [full_val._all_samples[i][1] for i in range(n)]

    val_idx, test_idx = train_test_split(
        list(range(n)),
        test_size=test_ratio,
        random_state=42,
        stratify=all_labels
    )

    val_ds = CageDroneRFDataset(
        val_images, target_size=target_size, label_mode=label_mode,
        indices=val_idx
    )
    test_ds = CageDroneRFDataset(
        val_images, target_size=target_size, label_mode=label_mode,
        indices=test_idx
    )

    return train_ds, val_ds, test_ds
