"""Summarize open-set results including OpenMax for thesis tables."""

import json
from pathlib import Path

DISPLAY = {
    "smallrf": "SmallRFNet",
    "resnet": "RFResNet",
    "transformer": "RFTransformer",
    "efficientnet": "RFEfficientNet",
    "ast": "RFAST",
    "conformer": "RFConformer",
    "cnn1d": "RFCNN1D",
}

METHODS = ["MSP", "Energy", "Mahalanobis", "OpenMax"]


def summarize(ds_name):
    path = Path(f"outputs/multiclass_evaluation/openset_multiclass/{ds_name}/openset_summary.json")
    if not path.exists():
        print(f"Missing: {path}")
        return

    with open(path) as f:
        summary = json.load(f)

    print(f"\n=== {ds_name} ===")
    for model_key, holdouts in summary.items():
        if not isinstance(holdouts, dict):
            continue
        dname = DISPLAY.get(model_key, model_key)
        print(f"\n  {dname}")
        for cls_name, methods in holdouts.items():
            if not isinstance(methods, dict):
                continue
            scores = []
            for m in METHODS:
                if m in methods and isinstance(methods[m], dict):
                    a = methods[m].get("auroc")
                    if a is not None:
                        scores.append(f"{m}={a:.3f}")
            print(f"    {cls_name:20s}: {'  '.join(scores)}")


def best_per_model(ds_name):
    """Best AUROC across all methods per (model, holdout)."""
    path = Path(f"outputs/multiclass_evaluation/openset_multiclass/{ds_name}/openset_summary.json")
    if not path.exists():
        return

    with open(path) as f:
        summary = json.load(f)

    print(f"\n=== {ds_name} BEST AUROC PER MODEL ===")
    for model_key, holdouts in summary.items():
        if not isinstance(holdouts, dict):
            continue
        dname = DISPLAY.get(model_key, model_key)
        aurocs = []
        for cls_name, methods in holdouts.items():
            if not isinstance(methods, dict):
                continue
            best = 0
            best_m = ""
            for m in METHODS:
                if m in methods and isinstance(methods[m], dict):
                    a = methods[m].get("auroc", 0)
                    if a > best:
                        best = a
                        best_m = m
            aurocs.append((cls_name, best, best_m))
        if aurocs:
            avg = sum(a[1] for a in aurocs) / len(aurocs)
            scores_str = " | ".join(f"{c}={a:.3f}({m})" for c, a, m in aurocs)
            print(f"  {dname:15s} avg={avg:.3f} | {scores_str}")


def main():
    for ds in ["DroneRF", "CageDroneRF", "RFUAV"]:
        summarize(ds)
    print("\n" + "=" * 70)
    for ds in ["DroneRF", "CageDroneRF", "RFUAV"]:
        best_per_model(ds)


if __name__ == "__main__":
    main()
