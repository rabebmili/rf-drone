"""Dataset pour le chargement de segments bruts de signaux RF (modèles 1D-CNN)."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.load_signal import load_dronerf_csv


class DroneRFRawDataset(Dataset):
    """Charge les segments bruts depuis les fichiers CSV, retourne un tenseur 1D [1, L]."""

    def __init__(self, csv_path, split="train", label_col="label_binary",
                 window_size=131072):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.label_col = label_col
        self.window_size = window_size

        # Cache des signaux pour éviter la relecture
        self._signal_cache = {}

    def _load_signal(self, file_path):
        if file_path not in self._signal_cache:
            self._signal_cache[file_path] = load_dronerf_csv(file_path)
        return self._signal_cache[file_path]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        start = int(row["start"])
        end = int(row["end"])

        signal = self._load_signal(file_path)
        segment = signal[start:end]

        # Assurer une longueur constante
        if len(segment) < self.window_size:
            segment = np.pad(segment, (0, self.window_size - len(segment)))
        segment = segment[:self.window_size]

        x = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # [1, L]
        y = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return x, y
