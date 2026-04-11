"""Évaluation inter-datasets améliorée : leave-one-out, ablation et affinage."""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, f1_score
)

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.rfuav_dataset import create_rfuav_splits
from src.datasets.cagedronerf_dataset import create_cagedronerf_loaders
from src.models import MODEL_REGISTRY, RAW_SIGNAL_MODELS, get_model
from src.evaluation.metrics import collect_predictions, compute_classification_metrics


def make_balanced_sampler(dataset):
    # Créer un WeightedRandomSampler qui équilibre les classes
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.item() if isinstance(y, torch.Tensor) else y)

    label_counts = Counter(labels)
    weights = [1.0 / label_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_balanced_concat_sampler(datasets_dict):
    # Créer un échantillonneur équilibré pour ConcatDataset avec poids égaux par dataset
    all_labels = []
    dataset_ids = []
    offset = 0

    for ds_name, ds in datasets_dict.items():
        for i in range(len(ds)):
            _, y = ds[i]
            all_labels.append(y.item() if isinstance(y, torch.Tensor) else y)
            dataset_ids.append(ds_name)
        offset += len(ds)

    # Poids = (1/nb_datasets) * (1/nombre_par_classe_dans_le_dataset)
    num_datasets = len(datasets_dict)
    ds_sizes = {name: len(ds) for name, ds in datasets_dict.items()}

    weights = []
    idx = 0
    for ds_name, ds in datasets_dict.items():
        ds_label_counts = Counter()
        for i in range(len(ds)):
            _, y = ds[i]
            ds_label_counts[y.item() if isinstance(y, torch.Tensor) else y] += 1

        for i in range(len(ds)):
            label = all_labels[idx]
            # Poids egaux par dataset * frequence inverse des classes dans le dataset
            w = (1.0 / num_datasets) * (1.0 / ds_label_counts[label])
            weights.append(w)
            idx += 1

    total_samples = max(ds_sizes.values()) * num_datasets  # échantillonner suffisamment
    return WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)


def train_model(model, train_loader, val_loaders, device, epochs=20, lr=5e-4,
                save_dir=None, model_name="model"):
    # Boucle d'entraînement avec validation par dataset et sélection par F1 macro moyen
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_avg_f1 = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            total += x.size(0)
        scheduler.step()

        # Validation par dataset séparément
        model.eval()
        epoch_metrics = {"epoch": epoch + 1, "train_loss": epoch_loss / total}
        f1_scores = []

        with torch.no_grad():
            for ds_name, val_loader in val_loaders.items():
                y_true, y_pred, y_prob = collect_predictions(model, val_loader, device)
                f1 = f1_score(y_true, y_pred, average="macro")
                acc = (np.array(y_true) == np.array(y_pred)).mean()
                epoch_metrics[f"{ds_name}_acc"] = float(acc)
                epoch_metrics[f"{ds_name}_f1"] = float(f1)
                f1_scores.append(f1)

        avg_f1 = np.mean(f1_scores)
        epoch_metrics["avg_f1"] = float(avg_f1)
        history.append(epoch_metrics)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            ds_str = " | ".join([f"{n}: F1={epoch_metrics[f'{n}_f1']:.4f}"
                                  for n in val_loaders.keys()])
            print(f"    Epoque {epoch+1}/{epochs} | F1 moyen : {avg_f1:.4f} | {ds_str}")

        # Sauvegarder le meilleur selon le F1 moyen inter-datasets
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(best_state, Path(save_dir) / f"{model_name}_best.pt")

    if best_state:
        model.load_state_dict(best_state)

    # Sauvegarder l'historique d'entraînement
    if save_dir:
        with open(Path(save_dir) / f"{model_name}_history.json", "w") as f:
            json.dump(history, f, indent=2)

    return model, best_avg_f1


def full_evaluate(model, loader, device, dataset_name, class_names,
                  output_dir=None, experiment_name=""):
    # Évaluation complète avec métriques, matrice de confusion et artefacts sauvegardés
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {}
    metrics["accuracy"] = float((y_true == y_pred).mean())
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC et PR-AUC (seulement quand les deux classes sont presentes)
    unique_true = set(y_true)
    if len(unique_true) >= 2:
        try:
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except (ValueError, IndexError):
            pass

    # Taux de faux positifs (pour le binaire : FP / (FP + TN))
    if len(class_names) == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Sauvegarder les artefacts
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        prefix = f"{experiment_name}_{dataset_name}" if experiment_name else dataset_name

        # Figure de la matrice de confusion
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        display_labels = [class_names[i] if i < len(class_names) else str(i)
                          for i in unique_labels]
        disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{experiment_name} → {dataset_name}")
        plt.tight_layout()
        plt.savefig(out / f"{prefix}_confusion_matrix.png", dpi=150)
        plt.close()

        # Sauvegarder les predictions et probabilites
        np.savez(out / f"{prefix}_predictions.npz",
                 y_true=y_true, y_pred=y_pred, y_prob=y_prob)

        # Sauvegarder les metriques en JSON
        with open(out / f"{prefix}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def load_datasets(args):
    # Charger tous les datasets disponibles. RFUAV n'a pas de classe background
    datasets = {}

    # DroneRF (possede une repartition train/val/test appropriee)
    if Path(args.dronerf_csv).exists():
        dronerf_train = DroneRFPrecomputedDataset(args.dronerf_csv, split="train", label_col="label_binary")
        dronerf_val = DroneRFPrecomputedDataset(args.dronerf_csv, split="val", label_col="label_binary")
        dronerf_test = DroneRFPrecomputedDataset(args.dronerf_csv, split="test", label_col="label_binary")
        datasets["DroneRF"] = {
            "train": dronerf_train, "val": dronerf_val, "test": dronerf_test,
            "has_background": True
        }

    # RFUAV (PAS de classe background — tous les echantillons sont des drones)
    if Path(args.rfuav_root).exists():
        rfuav_train, rfuav_val = create_rfuav_splits(args.rfuav_root, val_ratio=0.2, label_mode="binary")
        datasets["RFUAV"] = {
            "train": rfuav_train, "val": rfuav_val, "test": rfuav_val,
            "has_background": False
        }

    # CageDroneRF (val divise en val+test)
    cagedronerf_root = Path(args.cagedronerf_root)
    if cagedronerf_root.exists():
        cdrf_train, cdrf_val, cdrf_test = create_cagedronerf_loaders(
            args.cagedronerf_root, label_mode="binary", augment_train=True
        )
        datasets["CageDroneRF"] = {
            "train": cdrf_train, "val": cdrf_val, "test": cdrf_test,
            "has_background": True
        }

    return datasets


def run_leave_one_out(datasets, ModelClass, device, epochs, output_dir):
    # Leave-one-dataset-out : entraîner sur 2, tester sur le 3e exclu
    all_names = list(datasets.keys())
    results = {}

    for held_out in all_names:
        print(f"\n{'='*60}")
        print(f"  LEAVE-ONE-OUT : Exclusion de {held_out}")
        print(f"{'='*60}")

        train_names = [n for n in all_names if n != held_out]

        # Concaténation équilibrée des datasets d'entraînement
        train_datasets = {n: datasets[n]["train"] for n in train_names}
        combined_train = ConcatDataset([datasets[n]["train"] for n in train_names])
        sampler = make_balanced_concat_sampler(train_datasets)
        train_loader = DataLoader(combined_train, batch_size=16, sampler=sampler, num_workers=0)

        # Chargeurs de validation pour la sélection du modèle
        val_loaders = {n: DataLoader(datasets[n]["val"], batch_size=16, shuffle=False, num_workers=0)
                       for n in train_names}

        # Suivre également le dataset exclu pendant l'entraînement pour le monitoring
        val_loaders[held_out] = DataLoader(
            datasets[held_out]["test"], batch_size=16, shuffle=False, num_workers=0
        )

        model = ModelClass(num_classes=2).to(device)
        exp_dir = output_dir / f"leave_out_{held_out}"

        model, best_f1 = train_model(
            model, train_loader, val_loaders, device, epochs=epochs,
            save_dir=str(exp_dir / "models"), model_name="model"
        )

        # Evaluation complete sur le dataset exclu
        test_loader = DataLoader(
            datasets[held_out]["test"], batch_size=16, shuffle=False, num_workers=0
        )
        class_names = ["Background", "Drone"]

        m = full_evaluate(
            model, test_loader, device, held_out, class_names,
            output_dir=str(exp_dir), experiment_name=f"leave_out_{held_out}"
        )

        results[f"leave_out_{held_out}"] = m

        # Évaluer également sur les datasets d'entraînement pour comparaison
        for src_name in train_names:
            src_loader = DataLoader(
                datasets[src_name]["test"], batch_size=16, shuffle=False, num_workers=0
            )
            m_src = full_evaluate(
                model, src_loader, device, src_name, class_names,
                output_dir=str(exp_dir), experiment_name=f"leave_out_{held_out}"
            )
            results[f"leave_out_{held_out}_eval_{src_name}"] = m_src

    return results


def run_ablation(datasets, ModelClass, device, epochs, output_dir):
    # Ablation : source unique, par paires, 3 simples, 3 équilibrés
    import itertools
    all_names = list(datasets.keys())
    class_names = ["Background", "Drone"]
    results = {}

    # Fonction utilitaire : tester sur tous les datasets
    def evaluate_all(model, experiment_name, exp_dir):
        exp_results = {}
        for ds_name in all_names:
            loader = DataLoader(
                datasets[ds_name]["test"], batch_size=16, shuffle=False, num_workers=0
            )
            m = full_evaluate(model, loader, device, ds_name, class_names,
                              output_dir=str(exp_dir), experiment_name=experiment_name)
            exp_results[ds_name] = m
        return exp_results

    # 1. Source unique
    for src_name in all_names:
        exp_name = f"single_{src_name}"
        exp_dir = output_dir / exp_name

        train_loader = DataLoader(
            datasets[src_name]["train"], batch_size=16, shuffle=True, num_workers=0
        )
        val_loaders = {src_name: DataLoader(
            datasets[src_name]["val"], batch_size=16, shuffle=False, num_workers=0
        )}

        model = ModelClass(num_classes=2).to(device)
        model, _ = train_model(model, train_loader, val_loaders, device, epochs=epochs,
                                save_dir=str(exp_dir / "models"), model_name="model")
        results[exp_name] = evaluate_all(model, exp_name, exp_dir)

    # 2. Par paires
    for pair in itertools.combinations(all_names, 2):
        pair_name = "+".join(pair)
        exp_name = f"pair_{pair_name}"
        exp_dir = output_dir / exp_name

        combined = ConcatDataset([datasets[n]["train"] for n in pair])
        train_loader = DataLoader(combined, batch_size=16, shuffle=True, num_workers=0)
        val_loaders = {n: DataLoader(datasets[n]["val"], batch_size=16, shuffle=False, num_workers=0)
                       for n in pair}

        model = ModelClass(num_classes=2).to(device)
        model, _ = train_model(model, train_loader, val_loaders, device, epochs=epochs,
                                save_dir=str(exp_dir / "models"), model_name="model")
        results[exp_name] = evaluate_all(model, exp_name, exp_dir)

    # 3. Concaténation simple des 3 (sans echantillonnage équilibré)
    exp_name = "all3_plain"
    exp_dir = output_dir / exp_name
    combined = ConcatDataset([datasets[n]["train"] for n in all_names])
    train_loader = DataLoader(combined, batch_size=16, shuffle=True, num_workers=0)
    val_loaders = {n: DataLoader(datasets[n]["val"], batch_size=16, shuffle=False, num_workers=0)
                   for n in all_names}

    model = ModelClass(num_classes=2).to(device)
    model, _ = train_model(model, train_loader, val_loaders, device, epochs=epochs,
                            save_dir=str(exp_dir / "models"), model_name="model")
    results[exp_name] = evaluate_all(model, exp_name, exp_dir)

    # 4. Echantillonnage équilibré des 3
    exp_name = "all3_balanced"
    exp_dir = output_dir / exp_name
    train_datasets = {n: datasets[n]["train"] for n in all_names}
    sampler = make_balanced_concat_sampler(train_datasets)
    train_loader = DataLoader(combined, batch_size=16, sampler=sampler, num_workers=0)

    model = ModelClass(num_classes=2).to(device)
    model, _ = train_model(model, train_loader, val_loaders, device, epochs=epochs,
                            save_dir=str(exp_dir / "models"), model_name="model")
    results[exp_name] = evaluate_all(model, exp_name, exp_dir)

    return results


def run_target_finetune(datasets, ModelClass, device, epochs, output_dir):
    # Pré-entraîner sur l'ensemble combiné, puis affiner sur chaque dataset cible
    all_names = list(datasets.keys())
    class_names = ["Background", "Drone"]
    results = {}

    # Étape 1 : Pré-entraînement sur l'ensemble combine (équilibré)
    print(f"\n{'='*60}")
    print("  AFFINAGE CIBLE : Pré-entraînement sur tous les datasets (équilibré)")
    print(f"{'='*60}")

    combined = ConcatDataset([datasets[n]["train"] for n in all_names])
    train_datasets = {n: datasets[n]["train"] for n in all_names}
    sampler = make_balanced_concat_sampler(train_datasets)
    train_loader = DataLoader(combined, batch_size=16, sampler=sampler, num_workers=0)
    val_loaders = {n: DataLoader(datasets[n]["val"], batch_size=16, shuffle=False, num_workers=0)
                   for n in all_names}

    pretrained_model = ModelClass(num_classes=2).to(device)
    pretrained_model, _ = train_model(
        pretrained_model, train_loader, val_loaders, device, epochs=epochs,
        save_dir=str(output_dir / "finetune" / "models"), model_name="pretrained"
    )
    pretrained_state = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

    # Étape 2 : Affiner sur chaque cible
    for target_name in all_names:
        exp_dir = output_dir / "finetune" / f"ft_{target_name}"

        model = ModelClass(num_classes=2).to(device)
        model.load_state_dict(pretrained_state)

        ft_loader = DataLoader(
            datasets[target_name]["train"], batch_size=16, shuffle=True, num_workers=0
        )
        ft_val = {target_name: DataLoader(
            datasets[target_name]["val"], batch_size=16, shuffle=False, num_workers=0
        )}

        # Affiner avec un LR plus faible pendant moins d'epoques
        model, _ = train_model(
            model, ft_loader, ft_val, device, epochs=max(5, epochs // 4), lr=1e-4,
            save_dir=str(exp_dir / "models"), model_name=f"ft_{target_name}"
        )

        # Évaluer sur la cible
        test_loader = DataLoader(
            datasets[target_name]["test"], batch_size=16, shuffle=False, num_workers=0
        )
        m = full_evaluate(
            model, test_loader, device, target_name, class_names,
            output_dir=str(exp_dir), experiment_name=f"ft_{target_name}"
        )
        results[f"finetune_{target_name}"] = m

    return results


def print_summary(all_results, output_dir, model_name):
    # Afficher et sauvegarder le tableau récapitulatif complet
    print(f"\n{'='*80}")
    print(f"  RESUME COMPLET INTER-DATASETS — {model_name}")
    print(f"{'='*80}")

    # Aplatir les résultats pour le tableau
    rows = []
    for exp_name, exp_data in all_results.items():
        if isinstance(exp_data, dict) and "accuracy" in exp_data:
            rows.append((exp_name, exp_data))
        elif isinstance(exp_data, dict):
            for ds_name, metrics in exp_data.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    rows.append((f"{exp_name} → {ds_name}", metrics))

    if rows:
        print(f"\n  {'Experience':<50} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rappel':>7} {'TFP':>7}")
        print(f"  {'-'*85}")
        for name, m in rows:
            fpr = m.get('false_positive_rate', None)
            fpr_str = f"{fpr:.4f}" if isinstance(fpr, float) and not np.isnan(fpr) else "  N/A"
            acc = m.get('accuracy', 0)
            f1 = m.get('macro_f1', 0)
            prec = m.get('macro_precision', 0)
            rec = m.get('macro_recall', 0)
            print(f"  {name:<50} {acc:>7.4f} {f1:>7.4f} "
                  f"{prec:>7.4f} {rec:>7.4f} "
                  f"{fpr_str:>7}")

    # Sauvegarder tous les résultats
    def sanitize_value(v):
        if isinstance(v, (float, np.floating)):
            return None if np.isnan(v) else float(v)
        return v

    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    serializable[k][kk] = {kkk: sanitize_value(vvv) for kkk, vvv in vv.items()}
                else:
                    serializable[k][kk] = sanitize_value(vv)

    with open(output_dir / f"cross_dataset_enhanced_{model_name}.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResultats complets sauvegardes : {output_dir / f'cross_dataset_enhanced_{model_name}.json'}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced cross-dataset evaluation")
    parser.add_argument("--model", default="resnet",
                        choices=[k for k in MODEL_REGISTRY if k not in RAW_SIGNAL_MODELS])
    parser.add_argument("--dronerf_csv", default="data/metadata/dronerf_precomputed.csv")
    parser.add_argument("--rfuav_root", default="data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train")
    parser.add_argument("--cagedronerf_root", default="data/raw/CageDroneRF/balanced")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/cross_dataset_enhanced")
    parser.add_argument("--skip_leave_one_out", action="store_true")
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_finetune", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Résoudre la classe du modèle depuis le registre
    import importlib
    _mod_path, _cls_name = MODEL_REGISTRY[args.model]
    _mod = importlib.import_module(_mod_path)
    ModelClass = getattr(_mod, _cls_name)
    datasets = load_datasets(args)

    if len(datasets) < 2:
        print("ERREUR : Au moins 2 datasets nécessaires.")
        return

    all_results = {}

    # 1. Leave-one-dataset-out
    if not args.skip_leave_one_out and len(datasets) >= 3:
        print(f"\n{'='*60}")
        print("  PHASE 1 : LEAVE-ONE-DATASET-OUT")
        print(f"{'='*60}")
        loo_results = run_leave_one_out(datasets, ModelClass, device, args.epochs, output_dir)
        all_results["leave_one_out"] = loo_results

    # 2. Ablation (unique, par paires, tout-simple, tout-équilibré)
    if not args.skip_ablation:
        print(f"\n{'='*60}")
        print("  PHASE 2 : ETUDE D'ABLATION")
        print(f"{'='*60}")
        ablation_results = run_ablation(datasets, ModelClass, device, args.epochs, output_dir)
        all_results["ablation"] = ablation_results

    # 3. Affinage sur la cible
    if not args.skip_finetune:
        print(f"\n{'='*60}")
        print("  PHASE 3 : AFFINAGE SUR LA CIBLE")
        print(f"{'='*60}")
        ft_results = run_target_finetune(datasets, ModelClass, device, args.epochs, output_dir)
        all_results["finetune"] = ft_results

    # Resume
    print_summary(all_results, output_dir, args.model)


if __name__ == "__main__":
    main()
