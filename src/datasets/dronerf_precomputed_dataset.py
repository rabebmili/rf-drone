import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DroneRFPrecomputedDataset(Dataset):
    def __init__(self, csv_path, split="train", label_col="label_binary"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        spec = np.load(row["spec_path"]).astype(np.float32)
        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        y = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return x, y