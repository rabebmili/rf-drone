"""Robustesse, explicabilité, baselines et open-set sur le modèle binaire combiné."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.rfuav_dataset import create_rfuav_splits
from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
from src.models import get_model
from src.evaluation.robustness import run_robustness_evaluation
from src.evaluation.explainability import generate_gradcam_examples


# All spectrogram-based models (combined model is always spectrogram-based)
SPECTROGRAM_MODELS = ["smallrf", "resnet", "transformer", "efficientnet", "ast", "conformer"]

MODEL_DISPLAY_NAMES = {
    "smallrf": "SmallRFNet",
    "resnet": "RFResNet",
    "transformer": "RFTransformer",
    "efficientnet": "RFEfficientNet",
    "ast": "RFAST",
    "conformer": "RFConformer",
}


def load_combined_binary_data(dronerf_csv, rfuav_root, cagedronerf_root):
    datasets = {}

    if Path(dronerf_csv).exists():
        dronerf_test = DroneRFPrecomputedDataset(dronerf_csv, split="test", label_col="label_binary")
        datasets["DroneRF"] = dronerf_test

    if Path(rfuav_root).exists():
        _, rfuav_val = create_rfuav_splits(rfuav_root, val_ratio=0.2, label_mode="binary")
        datasets["RFUAV"] = rfuav_val

    if Path(cagedronerf_root).exists():
        _, _, cdrf_test = create_cagedronerf_loaders(cagedronerf_root, label_mode="binary")
        datasets["CageDroneRF"] = cdrf_test

    return datasets


def run_robustness_combined(model, datasets, device, output_dir, model_name="resnet"):
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION DE ROBUSTESSE — Modèle binaire combiné ({model_name})")
    print(f"{'='*60}")

    class_names = ["Background", "Drone"]
    out = Path(output_dir) / "robustness"

    for ds_name, ds in datasets.items():
        run_robustness_evaluation(
            model, ds, device,
            output_dir=str(out / ds_name),
            model_name=f"Combined_{MODEL_DISPLAY_NAMES.get(model_name, model_name)} on {ds_name}",
            class_names=class_names
        )


def run_explainability_combined(model, datasets, device, output_dir, model_name="resnet"):
    print(f"\n{'='*60}")
    print(f"  EXPLICABILITÉ (Grad-CAM) — Modèle binaire combiné ({model_name})")
    print(f"{'='*60}")

    class_names = ["Background", "Drone"]
    out = Path(output_dir) / "explainability"

    for ds_name, ds in datasets.items():
        try:
            generate_gradcam_examples(
                model, ds, device,
                model_name=model_name,
                class_names=class_names,
                output_dir=str(out / ds_name),
                n_per_class=5
            )
        except Exception as e:
            print(f"ERREUR : Grad-CAM échoué pour {ds_name}: {e}")


def run_baselines_combined(datasets, output_dir):
    print(f"\n{'='*60}")
    print("  BASELINES (SVM + RF) — Données binaires combinées")
    print(f"{'='*60}")

    from src.evaluation.feature_extraction import extract_features_from_dataset
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
    import numpy as np
    import joblib

    dronerf_csv = "data/metadata/dronerf_precomputed.csv"
    rfuav_root = "data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train"
    cagedronerf_root = "data/raw/CageDroneRF/balanced"

    train_datasets = []
    test_datasets = {}

    if Path(dronerf_csv).exists():
        train_ds = DroneRFPrecomputedDataset(dronerf_csv, split="train", label_col="label_binary")
        test_ds = DroneRFPrecomputedDataset(dronerf_csv, split="test", label_col="label_binary")
        train_datasets.append(train_ds)
        test_datasets["DroneRF"] = test_ds

    if Path(rfuav_root).exists():
        rfuav_train, rfuav_val = create_rfuav_splits(rfuav_root, val_ratio=0.2, label_mode="binary")
        train_datasets.append(rfuav_train)
        test_datasets["RFUAV"] = rfuav_val

    if Path(cagedronerf_root).exists():
        cdrf_train, _, cdrf_test = create_cagedronerf_loaders(cagedronerf_root, label_mode="binary")
        train_datasets.append(cdrf_train)
        test_datasets["CageDroneRF"] = cdrf_test

    combined_train = ConcatDataset(train_datasets)
    X_train, y_train = extract_features_from_dataset(combined_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    class_names = ["Background", "Drone"]
    out = Path(output_dir) / "baselines"
    out.mkdir(parents=True, exist_ok=True)

    for clf_name, clf in [
        ("SVM", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ("Random_Forest", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
    ]:
        print(f"\n  Entraînement {clf_name}...")
        clf.fit(X_train, y_train)

        joblib.dump(clf, out / f"combined_{clf_name.lower()}_model.joblib")
        joblib.dump(scaler, out / f"combined_{clf_name.lower()}_scaler.joblib")

        all_results = {}
        for ds_name, ds in test_datasets.items():
            X_test, y_test = extract_features_from_dataset(ds)
            X_test = scaler.transform(X_test)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            roc = None
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)
                try:
                    roc = roc_auc_score(y_test, y_prob[:, 1])
                except ValueError:
                    pass

            all_results[ds_name] = {"accuracy": acc, "macro_f1": f1, "roc_auc": roc}

        with open(out / f"combined_{clf_name.lower()}_results.json", "w") as f:
            json.dump(all_results, f, indent=2)


def run_openset_cagedronerf(device, output_dir, model_key="resnet"):
    """Évaluation open-set sur le modèle multiclasse CageDroneRF."""
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION OPEN-SET — CageDroneRF Multiclasse ({model_key})")
    print(f"{'='*60}")

    from src.evaluation.openset import run_openset_evaluation

    cagedronerf_root = "data/raw/CageDroneRF/balanced"
    if not Path(cagedronerf_root).exists():
        print("ERREUR : CageDroneRF non trouvé, open-set ignoré.")
        return

    cdrf_train, _, cdrf_test = create_cagedronerf_loaders(
        cagedronerf_root, label_mode="multiclass"
    )

    num_classes = cdrf_train.num_classes
    class_names = cdrf_train.get_class_names()

    weights_path = f"outputs/cagedronerf_{model_key}_multiclass/models/best_model.pt"
    if not Path(weights_path).exists():
        print(f"ERREUR : Modèle multiclasse absent à {weights_path}, ignoré.")
        return

    display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    model = get_model(model_key, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))

    train_loader = DataLoader(cdrf_train, batch_size=16, shuffle=False, num_workers=0)
    out = Path(output_dir) / "openset"

    holdout_classes = {
        "Skydio_2": class_names.index("Skydio_2"),
        "Parrot_Anafi": class_names.index("Parrot_Anafi"),
        "background": class_names.index("background"),
        "Hubsan_X4_Air": class_names.index("Hubsan_X4_Air"),
    }

    all_results = {}
    for cls_name, cls_id in holdout_classes.items():
        try:
            results = run_openset_evaluation(
                model, cdrf_test, device,
                holdout_class=cls_id,
                train_loader=train_loader,
                num_known_classes=num_classes,
                output_dir=str(out / f"holdout_{cls_name}")
            )
            all_results[cls_name] = results
        except Exception as e:
            print(f"ERREUR : Open-set échoué pour {cls_name}: {e}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "openset_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out / 'openset_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Combined model evaluation suite")
    parser.add_argument("--combined_model", default="outputs/cross_dataset_enhanced/all3_balanced/models/model_best.pt")
    parser.add_argument("--model", default="resnet", choices=list(MODEL_DISPLAY_NAMES.keys()),
                        help="Model architecture to use")
    parser.add_argument("--dronerf_csv", default="data/metadata/dronerf_precomputed.csv")
    parser.add_argument("--rfuav_root", default="data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train")
    parser.add_argument("--cagedronerf_root", default="data/raw/CageDroneRF/balanced")
    parser.add_argument("--output_dir", default="outputs/cross_dataset_enhanced/evaluation")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_explainability", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_openset", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.combined_model).exists():
        print(f"ERREUR : Modèle combiné non trouvé : {args.combined_model}")
        print("Exécuter d'abord l'évaluation inter-datasets : python -m src.evaluation.cross_dataset_enhanced --model resnet --epochs 20")
        return

    model = get_model(args.model, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.combined_model, weights_only=True, map_location=device))

    datasets = load_combined_binary_data(
        args.dronerf_csv, args.rfuav_root, args.cagedronerf_root
    )

    if not args.skip_robustness:
        run_robustness_combined(model, datasets, device, args.output_dir, model_name=args.model)

    if not args.skip_explainability:
        run_explainability_combined(model, datasets, device, args.output_dir, model_name=args.model)

    if not args.skip_baselines:
        run_baselines_combined(datasets, args.output_dir)

    if not args.skip_openset:
        run_openset_cagedronerf(device, args.output_dir, model_key=args.model)

    print(f"\nToutes les évaluations terminées. Résultats dans : {args.output_dir}/")


if __name__ == "__main__":
    main()
