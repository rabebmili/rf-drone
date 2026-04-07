"""Robustesse, explicabilité, baselines et open-set sur le modèle binaire combiné."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.rfuav_dataset import create_rfuav_splits
from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
from src.models.resnet_spectrogram import RFResNet
from src.models.cnn_spectrogram import SmallRFNet
from src.evaluation.robustness import run_robustness_evaluation
from src.evaluation.explainability import generate_gradcam_examples


def load_combined_binary_data(dronerf_csv, rfuav_root, cagedronerf_root):
    # Charger les données de test des 3 datasets pour l'évaluation binaire
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


def run_robustness_combined(model, datasets, device, output_dir):
    # Évaluation de robustesse du modèle combiné, par dataset
    print(f"\n{'='*60}")
    print("  ÉVALUATION DE ROBUSTESSE — Modèle binaire combiné")
    print(f"{'='*60}")

    class_names = ["Background", "Drone"]
    out = Path(output_dir) / "robustness"

    for ds_name, ds in datasets.items():
        run_robustness_evaluation(
            model, ds, device,
            output_dir=str(out / ds_name),
            model_name=f"Combined_ResNet on {ds_name}",
            class_names=class_names
        )


def run_explainability_combined(model, datasets, device, output_dir):
    # Exécuter Grad-CAM sur le modèle combiné, par dataset
    print(f"\n{'='*60}")
    print("  EXPLICABILITÉ (Grad-CAM) — Modèle binaire combiné")
    print(f"{'='*60}")

    class_names = ["Background", "Drone"]
    out = Path(output_dir) / "explainability"

    for ds_name, ds in datasets.items():
        try:
            generate_gradcam_examples(
                model, ds, device,
                model_name="resnet",
                class_names=class_names,
                output_dir=str(out / ds_name),
                n_per_class=5
            )
        except Exception as e:
            print(f"ERREUR : Grad-CAM échoué pour {ds_name}: {e}")


def run_baselines_combined(datasets, output_dir):
    # Entraîner SVM + RF sur les caractéristiques combinées de tous les datasets
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

    # On a aussi besoin des données d'entraînement pour les baselines
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

    # Combiner les données d'entraînement
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

        # Sauvegarder le modèle
        joblib.dump(clf, out / f"combined_{clf_name.lower()}_model.joblib")
        joblib.dump(scaler, out / f"combined_{clf_name.lower()}_scaler.joblib")

        # Tester sur chaque dataset séparément
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


def run_openset_cagedronerf(device, output_dir):
    # Évaluation open-set sur le modèle multiclasse CageDroneRF
    print(f"\n{'='*60}")
    print("  ÉVALUATION OPEN-SET — CageDroneRF Multiclasse")
    print(f"{'='*60}")

    from src.evaluation.openset import run_openset_evaluation

    cagedronerf_root = "data/raw/CageDroneRF/balanced"
    if not Path(cagedronerf_root).exists():
        print("ERREUR : CageDroneRF non trouvé, open-set ignoré.")
        return

    # Charger CageDroneRF multiclasse
    cdrf_train, _, cdrf_test = create_cagedronerf_loaders(
        cagedronerf_root, label_mode="multiclass"
    )

    num_classes = cdrf_train.num_classes
    class_names = cdrf_train.get_class_names()

    # Charger le meilleur modèle ResNet multiclasse CageDroneRF
    weights_path = "outputs/cagedronerf_resnet_multiclass/models/best_model.pt"
    if not Path(weights_path).exists():
        print(f"ERREUR : Modèle multiclasse absent à {weights_path}, ignoré.")
        return

    model = RFResNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))

    train_loader = DataLoader(cdrf_train, batch_size=16, shuffle=False, num_workers=0)
    out = Path(output_dir) / "openset"

    # Exclure plusieurs classes intéressantes
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

    # Sauvegarder le résumé
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "openset_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out / 'openset_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Combined model evaluation suite")
    parser.add_argument("--combined_model", default="outputs/cross_dataset_enhanced/all3_balanced/models/model_best.pt")
    parser.add_argument("--dronerf_csv", default="data/metadata/dronerf_precomputed.csv")
    parser.add_argument("--rfuav_root", default="data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train")
    parser.add_argument("--cagedronerf_root", default="data/raw/CageDroneRF/balanced")
    parser.add_argument("--output_dir", default="outputs/evaluation_combined_model")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_explainability", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_openset", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le modèle binaire combiné (ResNet entraîné sur les 3 datasets)
    if not Path(args.combined_model).exists():
        print(f"ERREUR : Modèle combiné non trouvé : {args.combined_model}")
        print("Exécuter d'abord l'évaluation inter-datasets : python -m src.evaluation.cross_dataset_enhanced --model resnet --epochs 20")
        return

    model = RFResNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.combined_model, weights_only=True, map_location=device))

    # Charger les données de test de tous les datasets
    datasets = load_combined_binary_data(
        args.dronerf_csv, args.rfuav_root, args.cagedronerf_root
    )

    # 1. Robustesse
    if not args.skip_robustness:
        run_robustness_combined(model, datasets, device, args.output_dir)

    # 2. Explicabilité
    if not args.skip_explainability:
        run_explainability_combined(model, datasets, device, args.output_dir)

    # 3. Baselines
    if not args.skip_baselines:
        run_baselines_combined(datasets, args.output_dir)

    # 4. Open-set (CageDroneRF multiclasse)
    if not args.skip_openset:
        run_openset_cagedronerf(device, args.output_dir)

    print(f"\nToutes les évaluations terminées. Résultats dans : {args.output_dir}/")


if __name__ == "__main__":
    main()
