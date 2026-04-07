"""Évaluer l'ensemble CNN+Transformer sur tous les datasets et tâches."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.ensemble import EnsembleCNNTransformer
from src.training.train_multimodel import load_dataset
from src.evaluation.metrics import full_evaluation


def evaluate_ensemble(dataset_name, task, cnn_name, tf_name, fusion="average"):
    # Évaluer l'ensemble sur une combinaison dataset/tâche
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Déterminer les chemins des poids
    if dataset_name == "dronerf":
        cnn_w = f"outputs/{cnn_name}_{task}/models/best_model.pt"
        tf_w = f"outputs/{tf_name}_{task}/models/best_model.pt"
    else:
        cnn_w = f"outputs/{dataset_name}_{cnn_name}_{task}/models/best_model.pt"
        tf_w = f"outputs/{dataset_name}_{tf_name}_{task}/models/best_model.pt"

    if not Path(cnn_w).exists() or not Path(tf_w).exists():
        print(f"  Ignoré {dataset_name}/{task}: poids non trouvés")
        print(f"    CNN: {cnn_w} exists={Path(cnn_w).exists()}")
        print(f"    TF:  {tf_w} exists={Path(tf_w).exists()}")
        return None

    # Charger le dataset
    train_ds, val_ds, test_ds, num_classes, class_names = load_dataset(
        dataset_name, task
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Construire l'ensemble
    model = EnsembleCNNTransformer(
        num_classes=num_classes,
        cnn_name=cnn_name,
        transformer_name=tf_name,
        fusion=fusion,
    )
    model.load_pretrained(cnn_w, tf_w, device=device)
    model = model.to(device)
    model.eval()

    # Évaluation complète
    metrics, _, _, _ = full_evaluation(
        model, test_loader, device, class_names,
        output_dir=None, model_name=f"ensemble_{cnn_name}+{tf_name}"
    )
    results = dict(metrics)
    results["model"] = f"ensemble_{cnn_name}+{tf_name}"
    results["fusion"] = fusion
    results["dataset"] = dataset_name
    results["task"] = task
    results["cnn_weights"] = cnn_w
    results["tf_weights"] = tf_w

    return results


def main():
    out_dir = Path("outputs/ensemble_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Évaluer sur tous les datasets/tâches
    configs = [
        ("dronerf", "binary"),
        ("dronerf", "multiclass"),
        ("cagedronerf", "binary"),
        ("cagedronerf", "multiclass"),
        ("rfuav", "multiclass"),
    ]

    cnn_name = "resnet"
    tf_name = "ast"

    for dataset, task in configs:
        tag = f"{dataset}_{task}"
        print(f"\n{'='*50}")
        print(f"  Ensemble {cnn_name}+{tf_name} / {dataset} / {task}")
        print(f"{'='*50}")

        result = evaluate_ensemble(dataset, task, cnn_name, tf_name, fusion="average")
        if result:
            all_results[tag] = result
            acc = result["accuracy"]
            f1 = result["macro_f1"]
            mcc = result["mcc"]
            print(f"  Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | MCC: {mcc:.4f}")

    # Sauvegarder tous les résultats
    with open(out_dir / "ensemble_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Afficher le résumé
    print(f"\n{'='*50}")
    print("  RÉSUMÉ DES RÉSULTATS D'ENSEMBLE")
    print(f"{'='*50}")
    for tag, r in all_results.items():
        print(f"  {tag}: acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} mcc={r['mcc']:.4f}")

    print(f"\nRésultats sauvegardés : {out_dir}/ensemble_results.json")


if __name__ == "__main__":
    main()
