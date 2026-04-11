"""Évaluation de robustesse multiclasse et open-set sur tous les datasets avec tous les modèles."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.dronerf_raw_dataset import DroneRFRawDataset
from src.datasets.rfuav_dataset import create_rfuav_splits
from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
from src.models import get_model
from src.evaluation.robustness import run_robustness_evaluation
from src.evaluation.openset import run_openset_evaluation


# Models available per dataset
SPECTROGRAM_MODELS = ["smallrf", "resnet", "transformer", "efficientnet", "ast", "conformer"]
DRONERF_MODELS = SPECTROGRAM_MODELS + ["cnn1d"]

MODEL_DISPLAY_NAMES = {
    "smallrf": "SmallRFNet",
    "resnet": "RFResNet",
    "transformer": "RFTransformer",
    "efficientnet": "RFEfficientNet",
    "ast": "RFAST",
    "conformer": "RFConformer",
    "cnn1d": "RFCNN1D",
}


def _model_path(ds_key, model_key, task="multiclass"):
    return f"outputs/{ds_key}_{model_key}_{task}/models/best_model.pt"


# ── Configuration des datasets ──────────────────────────────────

DATASET_CONFIGS = {
    "DroneRF": {
        "num_classes": 4,
        "class_names": ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"],
        "models": DRONERF_MODELS,
        "holdout_classes": {
            "AR Drone": 1,
            "Bepop Drone": 2,
            "Phantom Drone": 3,
        },
    },
    "CageDroneRF": {
        "num_classes": 27,
        "class_names": None,
        "models": SPECTROGRAM_MODELS,
        "holdout_classes": {
            "Skydio_2": None,
            "Parrot_Anafi": None,
            "background": None,
            "Hubsan_X4_Air": None,
            "DJI_Tello": None,
        },
    },
    "RFUAV": {
        "num_classes": 37,
        "class_names": None,
        "models": SPECTROGRAM_MODELS,
        "holdout_classes": None,
    },
}


def load_dataset(name, model_key="resnet"):
    if name == "DroneRF":
        if model_key == "cnn1d":
            csv_path = "data/metadata/dronerf_segments_split.csv"
            if not Path(csv_path).exists():
                return None, None
            train_ds = DroneRFRawDataset(csv_path, split="train", label_col="label_multiclass")
            test_ds = DroneRFRawDataset(csv_path, split="test", label_col="label_multiclass")
            return train_ds, test_ds
        else:
            csv_path = "data/metadata/dronerf_precomputed.csv"
            if not Path(csv_path).exists():
                return None, None
            train_ds = DroneRFPrecomputedDataset(csv_path, split="train", label_col="label_multiclass")
            test_ds = DroneRFPrecomputedDataset(csv_path, split="test", label_col="label_multiclass")
            return train_ds, test_ds

    elif name == "CageDroneRF":
        root = "data/raw/CageDroneRF/balanced"
        if not Path(root).exists():
            return None, None
        train_ds, _, test_ds = create_cagedronerf_loaders(root, label_mode="multiclass")
        return train_ds, test_ds

    elif name == "RFUAV":
        root = "data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train"
        if not Path(root).exists():
            return None, None
        train_ds, val_ds = create_rfuav_splits(root, val_ratio=0.2, label_mode="multiclass")
        return train_ds, val_ds

    return None, None


def run_robustness_multiclass(device, output_dir):
    print(f"\n{'='*60}")
    print("  ÉVALUATION DE ROBUSTESSE — Modèles multiclasses (tous les modèles)")
    print(f"{'='*60}")

    out_base = Path(output_dir) / "robustness_multiclass"
    all_results = {}

    for ds_name, config in DATASET_CONFIGS.items():
        ds_key = ds_name.lower()
        ds_results = {}

        for model_key in config["models"]:
            model_path = _model_path(ds_key, model_key)
            if not Path(model_path).exists():
                print(f"SKIP: {model_path} not found")
                continue

            train_ds, test_ds = load_dataset(ds_name, model_key)
            if test_ds is None:
                continue

            # Obtenir les noms de classes
            if config["class_names"] is not None:
                class_names = config["class_names"]
                num_classes = config["num_classes"]
            else:
                class_names = test_ds.get_class_names()
                num_classes = test_ds.num_classes

            display_name = MODEL_DISPLAY_NAMES[model_key]
            model = get_model(model_key, num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

            out_dir = str(out_base / ds_name / model_key)
            results = run_robustness_evaluation(
                model, test_ds, device,
                output_dir=out_dir,
                model_name=f"{display_name} multiclass on {ds_name}",
                class_names=class_names
            )

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

    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "robustness_multiclass_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_base / 'robustness_multiclass_summary.json'}")

    return all_results


def run_openset_multiclass(device, output_dir):
    print(f"\n{'='*60}")
    print("  ÉVALUATION OPEN-SET — Modèles multiclasses (tous les modèles)")
    print(f"{'='*60}")

    out_base = Path(output_dir) / "openset_multiclass"
    all_results = {}

    for ds_name, config in DATASET_CONFIGS.items():
        ds_key = ds_name.lower()
        ds_results = {}

        for model_key in config["models"]:
            model_path = _model_path(ds_key, model_key)
            if not Path(model_path).exists():
                print(f"SKIP: {model_path} not found")
                continue

            train_ds, test_ds = load_dataset(ds_name, model_key)
            if test_ds is None:
                continue

            if config["class_names"] is not None:
                class_names = config["class_names"]
                num_classes = config["num_classes"]
            else:
                class_names = test_ds.get_class_names()
                num_classes = test_ds.num_classes

            display_name = MODEL_DISPLAY_NAMES[model_key]
            model = get_model(model_key, num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

            # Déterminer les classes à exclure
            if config["holdout_classes"] is not None:
                holdout_classes = {}
                for cls_name, cls_id in config["holdout_classes"].items():
                    if cls_id is not None:
                        holdout_classes[cls_name] = cls_id
                    elif cls_name in class_names:
                        holdout_classes[cls_name] = class_names.index(cls_name)
                    else:
                        print(f"ATTENTION : classe {cls_name} non trouvée dans {ds_name}")
            else:
                holdout_classes = {}
                n = len(class_names)
                indices_to_try = [0, n // 4, n // 2, 3 * n // 4, n - 1]
                for idx in indices_to_try:
                    if idx < n:
                        holdout_classes[class_names[idx]] = idx

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
            model_results = {}

            for cls_name, cls_id in holdout_classes.items():
                try:
                    results = run_openset_evaluation(
                        model, test_ds, device,
                        holdout_class=cls_id,
                        train_loader=train_loader,
                        num_known_classes=num_classes,
                        output_dir=str(out_base / ds_name / model_key / f"holdout_{cls_name}")
                    )
                    model_results[cls_name] = results
                except Exception as e:
                    print(f"ERREUR : Open-set échoué pour {display_name}/{cls_name}: {e}")

            ds_results[model_key] = model_results

        all_results[ds_name] = ds_results

        # Sauvegarder le résumé par dataset
        ds_out = out_base / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)
        with open(ds_out / "openset_summary.json", "w") as f:
            json.dump(ds_results, f, indent=2, default=str)

    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "openset_multiclass_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_base / 'openset_multiclass_summary.json'}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multiclass robustness + open-set evaluation (all models)")
    parser.add_argument("--output_dir", default="outputs/multiclass_evaluation")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_openset", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_robustness:
        run_robustness_multiclass(device, args.output_dir)

    if not args.skip_openset:
        run_openset_multiclass(device, args.output_dir)

    print(f"\nToutes les évaluations multiclasses terminées. Résultats dans : {args.output_dir}/")


if __name__ == "__main__":
    main()
