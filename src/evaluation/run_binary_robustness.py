"""Évaluation de robustesse binaire sur DroneRF et CageDroneRF avec modèles RFResNet par dataset."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
from src.models.resnet_spectrogram import RFResNet
from src.evaluation.robustness import run_robustness_evaluation


DATASET_CONFIGS = {
    "DroneRF": {
        "model_path": "outputs/resnet_binary/models/best_model.pt",
        "class_names": ["Background", "Drone"],
    },
    "CageDroneRF": {
        "model_path": "outputs/cagedronerf_resnet_binary/models/best_model.pt",
        "class_names": ["Background/Non-drone", "Drone"],
    },
}


def load_test_dataset(name):
    if name == "DroneRF":
        csv_path = "data/metadata/dronerf_precomputed.csv"
        if not Path(csv_path).exists():
            return None
        return DroneRFPrecomputedDataset(csv_path, split="test", label_col="label_binary")
    elif name == "CageDroneRF":
        root = "data/raw/CageDroneRF/balanced"
        if not Path(root).exists():
            return None
        _, _, test_ds = create_cagedronerf_loaders(root, label_mode="binary")
        return test_ds
    return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_base = Path("outputs/robustness_single_dataset_binary")
    all_results = {}

    for ds_name, config in DATASET_CONFIGS.items():
        model_path = config["model_path"]
        if not Path(model_path).exists():
            continue

        test_ds = load_test_dataset(ds_name)
        if test_ds is None:
            continue

        model = RFResNet(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

        out_dir = str(out_base / ds_name)
        results = run_robustness_evaluation(
            model, test_ds, device,
            output_dir=out_dir,
            model_name=f"RFResNet binary on {ds_name}",
            class_names=config["class_names"]
        )

        # Sauvegarder en JSON
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        serializable = {}
        for snr_key, metrics in results.items():
            serializable[snr_key] = {
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        with open(Path(out_dir) / "robustness_results.json", "w") as f:
            json.dump(serializable, f, indent=2)

        all_results[ds_name] = serializable

    # Sauvegarder le résumé
    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "robustness_binary_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_base / 'robustness_binary_summary.json'}")


if __name__ == "__main__":
    main()
