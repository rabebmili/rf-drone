"""CLI pour l'analyse forensique intégrée combinant tous les composants du pipeline."""

import argparse
from pathlib import Path

from src.forensics.integrated_pipeline import ForensicPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Integrated forensic analysis of RF signal files"
    )
    # Requis
    parser.add_argument("--file", required=True, help="Path to raw signal CSV file")

    # Configuration du classificateur
    parser.add_argument("--classifier_model", default="resnet",
                        help="Classifier model name (from registry)")
    parser.add_argument("--task", default="multiclass",
                        choices=["binary", "multiclass"])
    parser.add_argument("--classifier_weights", default=None,
                        help="Path to classifier weights (auto-detected if not set)")

    # Configuration VAE
    parser.add_argument("--vae_weights", default=None,
                        help="Path to VAE weights (skip if not set)")
    parser.add_argument("--vae_latent_dim", type=int, default=32)
    parser.add_argument("--vae_threshold", type=float, default=0.1,
                        help="Reconstruction error threshold for anomaly")

    # Configuration Siamese
    parser.add_argument("--siamese_weights", default=None,
                        help="Path to Siamese network weights")
    parser.add_argument("--siamese_backbone", default="resnet",
                        help="Backbone model for Siamese encoder")
    parser.add_argument("--gallery", default=None,
                        help="Path to .npz gallery of known drone embeddings")

    # Configuration OpenMax
    parser.add_argument("--openmax_params", default=None,
                        help="Path to fitted OpenMax parameters (.pkl)")

    # Configuration générale
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--anomaly_threshold", type=float, default=0.7,
                        help="Confidence threshold for classification anomaly flagging")
    args = parser.parse_args()

    # Valeurs par défaut
    if args.task == "binary":
        num_classes = 2
        class_names = ["Background", "Drone"]
    else:
        num_classes = 4
        class_names = ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]

    if args.classifier_weights is None:
        args.classifier_weights = (
            f"outputs/{args.classifier_model}_{args.task}/models/best_model.pt"
        )

    if args.output_dir is None:
        file_stem = Path(args.file).stem
        args.output_dir = f"outputs/forensic_integrated/{file_stem}"

    # Construire la configuration du pipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "classifier_model": args.classifier_model,
        "classifier_task": args.task,
        "classifier_weights": args.classifier_weights,
        "num_classes": num_classes,
        "class_names": class_names,
        "vae_weights": args.vae_weights,
        "vae_latent_dim": args.vae_latent_dim,
        "vae_threshold": args.vae_threshold,
        "siamese_weights": args.siamese_weights,
        "siamese_backbone": args.siamese_backbone,
        "gallery_path": args.gallery,
        "openmax_params_path": args.openmax_params,
        "device": device,
        "anomaly_threshold": args.anomaly_threshold,
    }

    # Initialiser le pipeline
    pipeline = ForensicPipeline(config)

    # Analyser le fichier
    print(f"\nAnalyzing: {args.file}")
    timeline = pipeline.analyze_file(args.file)

    # Générer le rapport
    report = pipeline.generate_report(timeline, args.file, args.output_dir)

    # Afficher le résumé
    print(f"\n{'='*60}")
    print(f"  INTEGRATED FORENSIC REPORT")
    print(f"{'='*60}")

    cls_summary = report.get("classification_summary", {})
    if cls_summary.get("class_distribution"):
        print(f"  Total segments:    {report['report_metadata']['total_segments']}")
        print(f"  Drone segments:    {cls_summary['drone_segments']}")
        print(f"  Anomalous:         {cls_summary['anomalous_segments']}")
        print(f"  Avg confidence:    {cls_summary['average_confidence']}")
        print(f"  Class distribution:")
        for cls, count in cls_summary["class_distribution"].items():
            print(f"    {cls}: {count}")

    os_summary = report.get("openset_summary", {})
    if os_summary.get("unknown_segments", 0) > 0:
        print(f"\n  Unknown segments:  {os_summary['unknown_segments']} "
              f"({os_summary['unknown_rate']}%)")

    anom_summary = report.get("anomaly_summary", {})
    if anom_summary.get("vae_anomalous_segments", 0) > 0:
        print(f"  VAE anomalies:     {anom_summary['vae_anomalous_segments']}")
        print(f"  Mean recon error:  {anom_summary['mean_reconstruction_error']}")

    attr_summary = report.get("attribution_summary", {})
    if attr_summary.get("most_attributed_class"):
        print(f"\n  Most attributed:   {attr_summary['most_attributed_class']}")
        print(f"  Attribution distribution:")
        for cls, count in attr_summary.get("attribution_distribution", {}).items():
            print(f"    {cls}: {count}")

    components = report["report_metadata"]["components"]
    active = [k for k, v in components.items() if v]
    print(f"\n  Active components: {', '.join(active)}")
    print(f"{'='*60}")
    print(f"  Report: {args.output_dir}/integrated_forensic_report.json")
    print(f"  Timeline: {args.output_dir}/integrated_timeline.png")


if __name__ == "__main__":
    main()
