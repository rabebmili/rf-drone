"""Évaluation de robustesse binaire sur DroneRF et CageDroneRF avec tous les modèles."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.dronerf_raw_dataset import DroneRFRawDataset
from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
from src.models import get_model
from src.evaluation.robustness import run_robustness_evaluation


# Models available per dataset (CNN1D only for DroneRF)
SPECTROGRAM_MODELS = ["smallrf", "resnet", "transformer", "efficientnet", "ast", "conformer"]
DRONERF_MODELS = SPECTROGRAM_MODELS + ["cnn1d"]
CAGEDRONERF_MODELS = SPECTROGRAM_MODELS

DATASET_CONFIGS = {
    "DroneRF": {
        "models": DRONERF_MODELS,
        "class_names": ["Background", "Drone"],
    },
    "CageDroneRF": {
        "models": CAGEDRONERF_MODELS,
        "class_names": ["Background/Non-drone", "Drone"],
    },
}

MODEL_DISPLAY_NAMES = {
    "smallrf": "SmallRFNet",
    "resnet": "RFResNet",
    "transformer": "RFTransformer",
    "efficientnet": "RFEfficientNet",
    "ast": "RFAST",
    "conformer": "RFConformer",
    "cnn1d": "RFCNN1D",
}


def _model_path(ds_key, model_key, task="binary"):
    """Return path to trained model weights."""
    return f"outputs/{ds_key}_{model_key}_{task}/models/best_model.pt"


def load_test_dataset(name, model_key):
    if name == "DroneRF":
        if model_key == "cnn1d":
            csv_path = "data/metadata/dronerf_segments_split.csv"
            if not Path(csv_path).exists():
                return None
            return DroneRFRawDataset(csv_path, split="test", label_col="label_binary")
        else:
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

    out_base = Path("outputs/binary_evaluation/robustness_binary")
    all_results = {}

    for ds_name, config in DATASET_CONFIGS.items():
        ds_key = ds_name.lower()
        ds_results = {}

        for model_key in config["models"]:
            model_path = _model_path(ds_key, model_key, "binary")
            if not Path(model_path).exists():
                print(f"SKIP: {model_path} not found")
                continue

            test_ds = load_test_dataset(ds_name, model_key)
            if test_ds is None:
                continue

            display_name = MODEL_DISPLAY_NAMES[model_key]
            model = get_model(model_key, num_classes=2).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

            out_dir = str(out_base / ds_name / model_key)
            results = run_robustness_evaluation(
                model, test_ds, device,
                output_dir=out_dir,
                model_name=f"{display_name} binary on {ds_name}",
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

            ds_results[model_key] = serializable

        all_results[ds_name] = ds_results

    # Sauvegarder le résumé
    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "robustness_binary_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_base / 'robustness_binary_summary.json'}")


if __name__ == "__main__":
    main()
