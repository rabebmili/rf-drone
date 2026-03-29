"""Traditional ML baselines (SVM + Random Forest) on handcrafted spectrogram features."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
import joblib

from src.evaluation.feature_extraction import extract_features_from_dataset


# ── Dataset loaders ─────────────────────────────────────────────

def load_dronerf(task):
    """Load precomputed DroneRF dataset."""
    from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
    csv_path = "data/metadata/dronerf_precomputed.csv"
    label_col = "label_binary" if task == "binary" else "label_multiclass"

    train_ds = DroneRFPrecomputedDataset(csv_path, split="train", label_col=label_col)
    val_ds = DroneRFPrecomputedDataset(csv_path, split="val", label_col=label_col)
    test_ds = DroneRFPrecomputedDataset(csv_path, split="test", label_col=label_col)

    if task == "binary":
        class_names = ["Background", "Drone"]
    else:
        class_names = ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]

    return train_ds, val_ds, test_ds, class_names


def load_cagedronerf(task):
    """Load CageDroneRF dataset."""
    from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
    train_ds, val_ds, test_ds = create_cagedronerf_loaders(
        "data/raw/CageDroneRF/balanced", label_mode=task,
        augment_train=False
    )

    if task == "binary":
        class_names = ["Background/Non-drone", "Drone"]
    else:
        class_names = train_ds.get_class_names()

    return train_ds, val_ds, test_ds, class_names


def load_rfuav(task):
    """Load RFUAV dataset (multiclass only, no background class)."""
    from src.datasets.rfuav_dataset import create_rfuav_splits
    root = "data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train"

    # RFUAV has no test set — use val as test
    train_ds, val_ds = create_rfuav_splits(root, val_ratio=0.2, label_mode=task)
    test_ds = val_ds

    class_names = train_ds.get_class_names() if task == "multiclass" else ["Non-drone", "Drone"]

    return train_ds, val_ds, test_ds, class_names


DATASET_LOADERS = {
    "dronerf": load_dronerf,
    "cagedronerf": load_cagedronerf,
    "rfuav": load_rfuav,
}

# Valid dataset x task combinations
VALID_COMBOS = [
    ("dronerf", "binary"),
    ("dronerf", "multiclass"),
    ("cagedronerf", "binary"),
    ("cagedronerf", "multiclass"),
    ("rfuav", "multiclass"),  # no binary — RFUAV has no background class
]


# ── Training and evaluation ─────────────────────────────────────

def train_and_evaluate_baseline(clf, clf_name, X_train, y_train, X_val, y_val,
                                X_test, y_test, scaler, output_dir, class_names):
    """Train an sklearn classifier and evaluate on val/test sets."""
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  {clf_name} trained in {train_time:.1f}s")

    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")

        num_classes = len(np.unique(y_train))
        roc = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X)
            if num_classes == 2:
                roc = roc_auc_score(y, y_prob[:, 1])
            else:
                try:
                    roc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
                except ValueError:
                    pass

        results[split_name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "roc_auc": roc,
        }

    results["train_time_seconds"] = train_time

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / f"{clf_name.lower().replace(' ', '_')}_model.joblib"
    scaler_path = out / f"{clf_name.lower().replace(' ', '_')}_scaler.joblib"
    results_path = out / f"{clf_name.lower().replace(' ', '_')}_results.json"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
        else:
            serializable[k] = float(v) if isinstance(v, (np.floating, float)) else v
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  Model saved: {model_path}")
    print(f"  Results saved: {results_path}")

    return results


def run_single(dataset_name, task, output_dir=None):
    """Run baselines for a single dataset x task combination."""
    if (dataset_name, task) not in VALID_COMBOS:
        print(f"  Skipped: {dataset_name} x {task} is not a valid combination")
        return

    if output_dir is None:
        output_dir = f"outputs/baselines_{dataset_name}_{task}"

    print(f"\nBaselines: {dataset_name} x {task}")

    loader = DATASET_LOADERS[dataset_name]
    train_ds, val_ds, test_ds, class_names = loader(task)

    X_train, y_train = extract_features_from_dataset(train_ds)
    X_val, y_val = extract_features_from_dataset(val_ds)
    X_test, y_test = extract_features_from_dataset(test_ds)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    all_results = {}

    # SVM with RBF kernel
    svm_clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    all_results["SVM"] = train_and_evaluate_baseline(
        svm_clf, "SVM", X_train, y_train, X_val, y_val,
        X_test, y_test, scaler, output_dir, class_names
    )

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    all_results["Random Forest"] = train_and_evaluate_baseline(
        rf_clf, "Random Forest", X_train, y_train, X_val, y_val,
        X_test, y_test, scaler, output_dir, class_names
    )

    print(f"\n  {'Model':<20} {'Test Acc':>10} {'Test F1':>10} {'Test AUC':>10}")
    print(f"  {'-'*50}")
    for name, res in all_results.items():
        test_res = res["test"]
        auc_str = f"{test_res['roc_auc']:.4f}" if test_res['roc_auc'] else "N/A"
        print(f"  {name:<20} {test_res['accuracy']:>10.4f} {test_res['macro_f1']:>10.4f} {auc_str:>10}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifiers (SVM + RF)")
    parser.add_argument("--dataset", default="all",
                        choices=["dronerf", "cagedronerf", "rfuav", "all"])
    parser.add_argument("--task", default=None,
                        choices=["binary", "multiclass"],
                        help="Task (if omitted with --dataset all, runs all valid combos)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.dataset == "all":
        combos = VALID_COMBOS
    else:
        if args.task:
            combos = [(args.dataset, args.task)]
        else:
            combos = [(d, t) for d, t in VALID_COMBOS if d == args.dataset]

    for dataset_name, task in combos:
        out = args.output_dir or f"outputs/baselines_{dataset_name}_{task}"
        run_single(dataset_name, task, output_dir=out)


if __name__ == "__main__":
    main()
