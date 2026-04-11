"""Compute OpenMax scores only for all models × datasets × holdout classes.

Merges results into existing open-set summary JSON files (preserves MSP/Energy/Mahalanobis).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.dronerf_raw_dataset import DroneRFRawDataset
from src.datasets.rfuav_dataset import create_rfuav_splits
from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
from src.models import get_model
from src.evaluation.openset import (
    fit_openmax,
    compute_openmax_scores,
    evaluate_ood_detection,
    create_openset_split,
    _plot_ood_distributions,
    run_openset_evaluation,
)


SPECTROGRAM_MODELS = ["smallrf", "resnet", "transformer", "efficientnet", "ast", "conformer"]
DRONERF_MODELS = SPECTROGRAM_MODELS + ["cnn1d"]


DATASET_CONFIGS = {
    "DroneRF": {
        "num_classes": 4,
        "models": DRONERF_MODELS,
        "holdout_classes": {
            "AR Drone": 1,
            "Bepop Drone": 2,
            "Phantom Drone": 3,
        },
    },
    "CageDroneRF": {
        "num_classes": 27,
        "models": SPECTROGRAM_MODELS,
        "holdout_classes": [
            "Skydio_2",
            "Parrot_Anafi",
            "background",
            "Hubsan_X4_Air",
            "DJI_Tello",
        ],
    },
    "RFUAV": {
        "num_classes": 37,
        "models": SPECTROGRAM_MODELS,
        "holdout_classes": None,  # Will use indices
    },
}


def _model_path(ds_key, model_key):
    return f"outputs/{ds_key}_{model_key}_multiclass/models/best_model.pt"


def load_dataset(name, model_key="resnet"):
    if name == "DroneRF":
        if model_key == "cnn1d":
            csv_path = "data/metadata/dronerf_segments_split.csv"
            if not Path(csv_path).exists():
                return None, None
            train_ds = DroneRFRawDataset(csv_path, split="train", label_col="label_multiclass")
            test_ds = DroneRFRawDataset(csv_path, split="test", label_col="label_multiclass")
            return train_ds, test_ds
        csv_path = "data/metadata/dronerf_precomputed.csv"
        if not Path(csv_path).exists():
            return None, None
        train_ds = DroneRFPrecomputedDataset(csv_path, split="train", label_col="label_multiclass")
        test_ds = DroneRFPrecomputedDataset(csv_path, split="test", label_col="label_multiclass")
        return train_ds, test_ds

    if name == "CageDroneRF":
        root = "data/raw/CageDroneRF/balanced"
        if not Path(root).exists():
            return None, None
        train_ds, _, test_ds = create_cagedronerf_loaders(root, label_mode="multiclass")
        return train_ds, test_ds

    if name == "RFUAV":
        root = "data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train"
        if not Path(root).exists():
            return None, None
        train_ds, val_ds = create_rfuav_splits(root, val_ratio=0.2, label_mode="multiclass")
        return train_ds, val_ds

    return None, None


def compute_openmax_for_holdout(model, train_loader, test_ds, holdout_idx, num_classes, device,
                                output_dir=None):
    """Fit OpenMax and compute AUROC for one holdout class."""
    mavs, weibull_params = fit_openmax(model, train_loader, device, num_classes)

    known_idx, unknown_idx = create_openset_split(test_ds, holdout_idx)
    if not unknown_idx or not known_idx:
        return None

    known_loader = DataLoader(Subset(test_ds, known_idx), batch_size=16, shuffle=False, num_workers=0)
    unknown_loader = DataLoader(Subset(test_ds, unknown_idx), batch_size=16, shuffle=False, num_workers=0)

    in_scores, _ = compute_openmax_scores(model, known_loader, device, mavs, weibull_params, num_classes)
    ood_scores, _ = compute_openmax_scores(model, unknown_loader, device, mavs, weibull_params, num_classes)

    if output_dir:
        _plot_ood_distributions(in_scores, ood_scores, "OpenMax Score", output_dir, "openmax_distribution.png")

    return evaluate_ood_detection(in_scores, ood_scores, "OpenMax")


def merge_openmax_into_summary(ds_name, results):
    """Merge OpenMax results into the existing openset_summary.json."""
    summary_path = Path(f"outputs/multiclass_evaluation/openset_multiclass/{ds_name}/openset_summary.json")
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    for model_key, holdouts in results.items():
        if model_key not in summary:
            summary[model_key] = {}
        for cls_name, openmax_res in holdouts.items():
            if cls_name not in summary[model_key]:
                summary[model_key][cls_name] = {}
            if openmax_res is not None:
                summary[model_key][cls_name]["OpenMax"] = openmax_res

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Merged into {summary_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ds_name, config in DATASET_CONFIGS.items():
        ds_key = ds_name.lower()
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        ds_results = {}

        for model_key in config["models"]:
            model_path = _model_path(ds_key, model_key)
            if not Path(model_path).exists():
                continue

            print(f"\n  --- {model_key} ---")
            train_ds, test_ds = load_dataset(ds_name, model_key)
            if test_ds is None:
                continue

            num_classes = config["num_classes"]
            if hasattr(test_ds, "num_classes"):
                num_classes = test_ds.num_classes

            class_names = None
            if hasattr(test_ds, "get_class_names"):
                class_names = test_ds.get_class_names()

            # Determine holdout classes
            if isinstance(config["holdout_classes"], dict):
                holdouts = config["holdout_classes"]
            elif isinstance(config["holdout_classes"], list):
                holdouts = {}
                for name in config["holdout_classes"]:
                    if class_names and name in class_names:
                        holdouts[name] = class_names.index(name)
            else:
                # RFUAV: pick by index
                holdouts = {}
                if class_names:
                    n = len(class_names)
                    for idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
                        if idx < n:
                            holdouts[class_names[idx]] = idx

            model = get_model(model_key, num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            model.eval()

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)

            model_results = {}
            for cls_name, cls_id in holdouts.items():
                # Keep spaces as-is to match existing MSP/Energy plot folders
                safe_cls = cls_name.replace("/", "_")
                out_dir = Path(
                    f"outputs/multiclass_evaluation/openset_multiclass/{ds_name}"
                    f"/{model_key}/holdout_{safe_cls}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    res = compute_openmax_for_holdout(
                        model, train_loader, test_ds, cls_id, num_classes, device,
                        output_dir=out_dir,
                    )
                    if res is not None:
                        model_results[cls_name] = res
                        print(f"    {cls_name:20s}: AUROC={res['auroc']:.3f}")
                except Exception as e:
                    import traceback
                    print(f"    {cls_name}: FAILED - {e}")
                    traceback.print_exc()

            ds_results[model_key] = model_results

        merge_openmax_into_summary(ds_name, ds_results)

    _fill_missing_entries(device)


def _fill_missing_entries(device: str) -> None:
    """Fill known gaps where run_multiclass_eval.py produced incomplete results.

    Currently covers: RFEfficientNet on CageDroneRF (Hubsan_X4_Air, DJI_Tello)
    — these two holdout classes were missing MSP/Energy/Mahalanobis scores.
    Runs all four open-set methods via run_openset_evaluation and patches the
    existing openset_summary.json in-place.
    """
    DATASET_ROOT = "data/raw/CageDroneRF/balanced"
    MODEL_PATH   = "outputs/cagedronerf_efficientnet_multiclass/models/best_model.pt"
    SUMMARY_PATH = Path(
        "outputs/multiclass_evaluation/openset_multiclass/CageDroneRF/openset_summary.json"
    )
    OUT_BASE     = Path(
        "outputs/multiclass_evaluation/openset_multiclass/CageDroneRF/efficientnet"
    )
    MISSING      = ["Hubsan_X4_Air", "DJI_Tello"]

    if not Path(DATASET_ROOT).exists() or not Path(MODEL_PATH).exists():
        return
    if not SUMMARY_PATH.exists():
        return

    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    existing = summary.get("efficientnet", {})
    targets = [m for m in MISSING if m not in existing or not existing[m].get("MSP")]
    if not targets:
        print("\n[fill_missing] EfficientNet/CageDroneRF entries already complete, skipping.")
        return

    print(f"\n[fill_missing] Filling EfficientNet/CageDroneRF gaps: {targets}")
    train_ds, _, test_ds = create_cagedronerf_loaders(DATASET_ROOT, label_mode="multiclass")
    class_names = test_ds.get_class_names()
    num_classes  = test_ds.num_classes

    model = get_model("efficientnet", num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)

    for cls_name in targets:
        if cls_name not in class_names:
            print(f"  WARNING: '{cls_name}' not in class list, skipping")
            continue
        cls_id  = class_names.index(cls_name)
        out_dir = str(OUT_BASE / f"holdout_{cls_name}")
        print(f"  Running: EfficientNet / holdout={cls_name} (id={cls_id})")
        results = run_openset_evaluation(
            model, test_ds, device,
            holdout_class=cls_id,
            train_loader=train_loader,
            num_known_classes=num_classes,
            output_dir=out_dir,
        )
        if "efficientnet" not in summary:
            summary["efficientnet"] = {}
        summary["efficientnet"][cls_name] = results
        print(
            f"    MSP={results.get('MSP', {}).get('auroc', 'N/A'):.4f}  "
            f"Energy={results.get('Energy', {}).get('auroc', 'N/A'):.4f}  "
            f"Mahl={results.get('Mahalanobis', {}).get('auroc', 'N/A'):.4f}  "
            f"OpenMax={results.get('OpenMax', {}).get('auroc', 'N/A'):.4f}"
        )

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fill_missing] Saved updated summary → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
