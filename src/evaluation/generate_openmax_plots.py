"""Generate OpenMax distribution plots for selected (model, dataset, holdout) cases.

Targets only the cases highlighted in the thesis (best + worst + most interesting).
Much faster than re-running the full run_openmax_only.py.
"""

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
    create_openset_split,
    _plot_ood_distributions,
)


# Selected cases: (ds_name, model_key, holdout_name, holdout_idx, num_classes)
# Chosen to illustrate OpenMax best / worst / most interesting in the thesis
CASES = [
    # Best: RFCNN1D on DroneRF AR Drone (OpenMax=0.883, best single value)
    ("DroneRF", "cnn1d", "AR Drone", 1, 4),
    # Second best on DroneRF: SmallRFNet AR Drone (OpenMax=0.766)
    ("DroneRF", "smallrf", "AR Drone", 1, 4),
    # OpenMax wins on a hard CageDroneRF case: RFConformer Hubsan_X4_Air (0.570)
    ("CageDroneRF", "conformer", "Hubsan_X4_Air", None, 27),
    # OpenMax fails badly: SmallRFNet Phantom Drone (0.146 vs 0.861 Mahal.)
    ("DroneRF", "smallrf", "Phantom Drone", 3, 4),
    # RFUAV: RFResNet JR PROPO XG14 (OpenMax=0.603, only OpenMax win on RFUAV)
    ("RFUAV", "resnet", "JR PROPO XG14", None, 37),
]


def _model_path(ds_key, model_key):
    return f"outputs/{ds_key}_{model_key}_multiclass/models/best_model.pt"


def load_dataset(ds_name, model_key):
    if ds_name == "DroneRF":
        if model_key == "cnn1d":
            csv = "data/metadata/dronerf_segments_split.csv"
            if not Path(csv).exists():
                return None, None
            return (
                DroneRFRawDataset(csv, split="train", label_col="label_multiclass"),
                DroneRFRawDataset(csv, split="test",  label_col="label_multiclass"),
            )
        csv = "data/metadata/dronerf_precomputed.csv"
        if not Path(csv).exists():
            return None, None
        return (
            DroneRFPrecomputedDataset(csv, split="train", label_col="label_multiclass"),
            DroneRFPrecomputedDataset(csv, split="test",  label_col="label_multiclass"),
        )
    if ds_name == "CageDroneRF":
        root = "data/raw/CageDroneRF/balanced"
        if not Path(root).exists():
            return None, None
        train_ds, _, test_ds = create_cagedronerf_loaders(root, label_mode="multiclass")
        return train_ds, test_ds
    if ds_name == "RFUAV":
        root = "data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train"
        if not Path(root).exists():
            return None, None
        return create_rfuav_splits(root, val_ratio=0.2, label_mode="multiclass")
    return None, None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for ds_name, model_key, cls_name, cls_id, num_classes in CASES:
        ds_key = ds_name.lower()
        model_path = _model_path(ds_key, model_key)
        if not Path(model_path).exists():
            print(f"[SKIP] {model_path} not found")
            continue

        print(f"--- {ds_name} / {model_key} / {cls_name} ---")
        train_ds, test_ds = load_dataset(ds_name, model_key)
        if test_ds is None:
            print("  dataset not found, skip")
            continue

        # Resolve cls_id from class names if needed
        if cls_id is None:
            if hasattr(test_ds, "get_class_names"):
                names = test_ds.get_class_names()
                if cls_name in names:
                    cls_id = names.index(cls_name)
            if cls_id is None:
                print(f"  Cannot resolve class index for '{cls_name}', skip")
                continue

        actual_classes = num_classes
        if hasattr(test_ds, "num_classes"):
            actual_classes = test_ds.num_classes

        model = get_model(model_key, num_classes=actual_classes).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)

        mavs, weibull_params = fit_openmax(model, train_loader, device, actual_classes)

        known_idx, unknown_idx = create_openset_split(test_ds, cls_id)
        if not known_idx or not unknown_idx:
            print("  empty split, skip")
            continue

        known_loader  = DataLoader(Subset(test_ds, known_idx),  batch_size=16, shuffle=False, num_workers=0)
        unknown_loader = DataLoader(Subset(test_ds, unknown_idx), batch_size=16, shuffle=False, num_workers=0)

        in_scores, _  = compute_openmax_scores(model, known_loader,  device, mavs, weibull_params, actual_classes)
        ood_scores, _ = compute_openmax_scores(model, unknown_loader, device, mavs, weibull_params, actual_classes)

        safe_cls = cls_name.replace("/", "_")
        out_dir = Path(
            f"outputs/multiclass_evaluation/openset_multiclass/{ds_name}"
            f"/{model_key}/holdout_{safe_cls}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        _plot_ood_distributions(in_scores, ood_scores, "OpenMax Score", out_dir, "openmax_distribution.png")

        from sklearn.metrics import roc_auc_score
        import numpy as np
        labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(ood_scores))])
        scores = np.concatenate([in_scores, ood_scores])
        auroc = roc_auc_score(labels, scores)
        print(f"  AUROC={auroc:.3f}  plot → {out_dir}/openmax_distribution.png")


if __name__ == "__main__":
    main()
