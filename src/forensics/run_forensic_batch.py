"""Analyse forensique par lots sur plusieurs fichiers signal RF avec résumé global."""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import MODEL_REGISTRY, RAW_SIGNAL_MODELS, get_model
from src.forensics.timeline import (
    analyze_signal_file,
    generate_forensic_report,
    plot_forensic_timeline,
)


def collect_csv_files(folder, recursive=False):
    # Collecte tous les fichiers CSV de signaux d'un dossier
    folder = Path(folder)
    if recursive:
        files = sorted(folder.rglob("*.csv"))
    else:
        files = sorted(folder.glob("*.csv"))
    return files


def plot_global_summary(all_file_results, output_dir, class_names):
    # Génère les graphiques de résumé global sur tous les fichiers analysés
    out = Path(output_dir)

    # ── 1. Distribution des classes sur tous les fichiers ──
    global_class_counts = {}
    for result in all_file_results:
        for cls, count in result["class_distribution"].items():
            global_class_counts[cls] = global_class_counts.get(cls, 0) + count

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_map = {
        "Background": "#607D8B", "AR Drone": "#2196F3",
        "Bepop Drone": "#4CAF50", "Phantom Drone": "#FF9800",
        "Drone": "#E53935",
    }
    labels = list(global_class_counts.keys())
    values = list(global_class_counts.values())
    bar_colors = [colors_map.get(l, "#9E9E9E") for l in labels]
    ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Nombre de segments", fontsize=12)
    ax.set_title("Distribution globale des classes\n(tous les fichiers analysés)",
                 fontsize=14, fontweight="bold")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", fontsize=10,
                fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "global_class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── 2. Distribution de confiance sur tous les fichiers ──
    all_confidences = []
    for result in all_file_results:
        all_confidences.extend(result["confidences"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_confidences, bins=50, color="#2196F3", edgecolor="white",
            alpha=0.85)
    ax.axvline(x=0.7, color="red", linestyle="--", lw=2, label="Seuil d'anomalie (0.7)")
    ax.set_xlabel("Score de confiance", fontsize=12)
    ax.set_ylabel("Nombre de segments", fontsize=12)
    ax.set_title("Distribution des scores de confiance\n(tous les segments)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "global_confidence_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── 3. Résumé par fichier ──
    fig, ax = plt.subplots(figsize=(14, max(4, len(all_file_results) * 0.35 + 1)))
    file_names = [r["file_name"] for r in all_file_results]
    avg_confs = [r["avg_confidence"] for r in all_file_results]
    anomaly_counts = [r["anomalous_count"] for r in all_file_results]
    drone_pcts = [r["drone_pct"] for r in all_file_results]

    y_pos = range(len(file_names))
    bar_colors_conf = ["#F44336" if c < 0.7 else "#4CAF50" if c > 0.9
                       else "#FF9800" for c in avg_confs]
    ax.barh(y_pos, avg_confs, color=bar_colors_conf, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(file_names, fontsize=8)
    ax.set_xlabel("Confiance moyenne", fontsize=12)
    ax.set_title("Confiance moyenne par fichier analysé",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.7, color="red", linestyle="--", lw=1.5, alpha=0.7)
    for i, (conf, anom) in enumerate(zip(avg_confs, anomaly_counts)):
        label = f"{conf:.3f}"
        if anom > 0:
            label += f" ({anom} anomalies)"
        ax.text(conf + 0.01, i, label, va="center", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out / "global_per_file_confidence.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── 4. Taux de détection drone par dossier (si récursif) ──
    folder_stats = {}
    for r in all_file_results:
        folder = r.get("folder", "unknown")
        if folder not in folder_stats:
            folder_stats[folder] = {"total": 0, "drone": 0}
        folder_stats[folder]["total"] += r["total_segments"]
        folder_stats[folder]["drone"] += r["drone_count"]

    if len(folder_stats) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        folders = list(folder_stats.keys())
        drone_rates = [folder_stats[f]["drone"] / max(folder_stats[f]["total"], 1) * 100
                       for f in folders]
        fc = [colors_map.get(f.replace(" drone", " Drone")
                             .replace("Background RF activites", "Background"),
                             "#9E9E9E") for f in folders]
        ax.bar(folders, drone_rates, color=fc, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Taux de détection drone (%)", fontsize=12)
        ax.set_title("Taux de détection par catégorie de source",
                     fontsize=14, fontweight="bold")
        for i, v in enumerate(drone_rates):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out / "global_detection_rate_by_folder.png",
                    dpi=200, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Batch forensic analysis on multiple RF signal files")
    parser.add_argument("--folder", required=True,
                        help="Folder containing signal CSV files")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search subfolders")
    parser.add_argument("--model", default="resnet",
                        choices=[k for k in MODEL_REGISTRY if k not in RAW_SIGNAL_MODELS])
    parser.add_argument("--task", default="multiclass",
                        choices=["binary", "multiclass"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--anomaly_threshold", type=float, default=0.7)
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max number of files to analyze (for quick test)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 2 if args.task == "binary" else 4
    class_names = (
        ["Background", "Drone"] if args.task == "binary"
        else ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]
    )

    # Charger le modèle
    if args.weights is None:
        args.weights = f"outputs/dronerf_{args.model}_{args.task}/models/best_model.pt"

    model = get_model(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(
        torch.load(args.weights, weights_only=True, map_location=device))
    model.eval()

    # Collecter les fichiers
    csv_files = collect_csv_files(args.folder, recursive=args.recursive)
    if args.max_files:
        csv_files = csv_files[:args.max_files]

    if not csv_files:
        print(f"ERROR: No CSV files found in {args.folder}")
        return

    # Répertoire de sortie
    if args.output_dir is None:
        args.output_dir = f"outputs/forensic_batch/{args.model}_{args.task}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Analyser chaque fichier
    all_file_results = []
    all_timelines = {}

    for file_idx, csv_file in enumerate(csv_files):
        file_stem = csv_file.stem
        folder_name = csv_file.parent.name

        try:
            timeline = analyze_signal_file(
                model, str(csv_file), device,
                class_names=class_names,
                anomaly_threshold=args.anomaly_threshold,
            )
        except Exception as e:
            print(f"ERROR: {csv_file.name}: {e}")
            continue

        # Statistiques par fichier
        class_dist = {}
        confidences = []
        anomalous_count = 0
        drone_count = 0
        for entry in timeline:
            label = entry["predicted_label"]
            class_dist[label] = class_dist.get(label, 0) + 1
            confidences.append(entry["confidence"])
            if entry["is_anomalous"]:
                anomalous_count += 1
            if entry["predicted_class"] != 0:
                drone_count += 1

        avg_conf = np.mean(confidences) if confidences else 0.0
        total = len(timeline)
        drone_pct = drone_count / total * 100 if total > 0 else 0.0

        file_result = {
            "file_path": str(csv_file),
            "file_name": f"{folder_name}/{file_stem}",
            "folder": folder_name,
            "total_segments": total,
            "drone_count": drone_count,
            "drone_pct": round(drone_pct, 2),
            "anomalous_count": anomalous_count,
            "avg_confidence": round(float(avg_conf), 4),
            "class_distribution": class_dist,
            "confidences": confidences,
        }
        all_file_results.append(file_result)
        all_timelines[str(csv_file)] = timeline

        # Sauvegarder le graphique de timeline individuel
        file_out = out_dir / "per_file" / folder_name
        file_out.mkdir(parents=True, exist_ok=True)
        plot_forensic_timeline(
            timeline,
            output_path=str(file_out / f"{file_stem}_timeline.png"),
            title=f"{folder_name}/{file_stem}",
        )

        # Sauvegarder le rapport individuel
        generate_forensic_report(
            timeline, str(csv_file),
            output_path=str(file_out / f"{file_stem}_report.json"),
            class_names=class_names,
        )

    if not all_file_results:
        print("ERROR: No files analyzed successfully.")
        return
    plot_global_summary(all_file_results, out_dir, class_names)

    # ── Résumé global JSON ──
    total_segments = sum(r["total_segments"] for r in all_file_results)
    total_drone = sum(r["drone_count"] for r in all_file_results)
    total_anomalous = sum(r["anomalous_count"] for r in all_file_results)
    all_confs = []
    for r in all_file_results:
        all_confs.extend(r["confidences"])

    global_class_dist = {}
    for r in all_file_results:
        for cls, cnt in r["class_distribution"].items():
            global_class_dist[cls] = global_class_dist.get(cls, 0) + cnt

    # Fichiers avec le plus d'anomalies
    anomalous_files = sorted(
        [r for r in all_file_results if r["anomalous_count"] > 0],
        key=lambda r: r["anomalous_count"], reverse=True
    )

    # Fichiers avec la confiance la plus faible
    low_conf_files = sorted(all_file_results, key=lambda r: r["avg_confidence"])[:10]

    # Retirer la liste de confiances par fichier du JSON (trop volumineux)
    results_for_json = []
    for r in all_file_results:
        r_copy = {k: v for k, v in r.items() if k != "confidences"}
        results_for_json.append(r_copy)

    global_report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": args.model,
            "task": args.task,
            "source_folder": str(args.folder),
            "recursive": args.recursive,
            "total_files_analyzed": len(all_file_results),
            "total_segments": total_segments,
            "framework": "RF Drone Forensics Pipeline",
        },
        "global_summary": {
            "total_segments": total_segments,
            "total_drone_segments": total_drone,
            "total_anomalous_segments": total_anomalous,
            "drone_detection_rate": round(total_drone / max(total_segments, 1) * 100, 2),
            "average_confidence": round(float(np.mean(all_confs)), 4),
            "min_confidence": round(float(np.min(all_confs)), 4),
            "max_confidence": round(float(np.max(all_confs)), 4),
            "class_distribution": global_class_dist,
        },
        "anomalous_files": [
            {"file": r["file_name"], "anomalies": r["anomalous_count"],
             "avg_confidence": r["avg_confidence"]}
            for r in anomalous_files[:20]
        ],
        "lowest_confidence_files": [
            {"file": r["file_name"], "avg_confidence": r["avg_confidence"],
             "drone_pct": r["drone_pct"]}
            for r in low_conf_files
        ],
        "per_file_results": results_for_json,
    }

    report_path = out_dir / "global_forensic_report.json"
    with open(report_path, "w") as f:
        json.dump(global_report, f, indent=2)

    # ── Affichage du résumé ──
    print(f"\n{'='*60}")
    print(f"  RAPPORT FORENSIQUE GLOBAL")
    print(f"{'='*60}")
    print(f"  Modèle          : {args.model} ({args.task})")
    print(f"  Fichiers         : {len(all_file_results)}")
    print(f"  Segments totaux  : {total_segments}")
    print(f"  Segments drone   : {total_drone} "
          f"({total_drone/max(total_segments,1)*100:.1f}%)")
    print(f"  Segments anomaux : {total_anomalous}")
    print(f"  Confiance moy.   : {np.mean(all_confs):.4f}")
    print(f"\n  Distribution des classes :")
    for cls, cnt in sorted(global_class_dist.items()):
        pct = cnt / total_segments * 100
        print(f"    {cls:<18} : {cnt:>5} segments ({pct:.1f}%)")
    if anomalous_files:
        print(f"\n  Fichiers avec anomalies :")
        for r in anomalous_files[:5]:
            print(f"    {r['file_name']:<40} : {r['anomalous_count']} anomalies")
    print(f"{'='*60}")
    print(f"\n  Rapport sauvegardé : {report_path}")
    print(f"  Graphiques dans   : {out_dir}/")


if __name__ == "__main__":
    main()
