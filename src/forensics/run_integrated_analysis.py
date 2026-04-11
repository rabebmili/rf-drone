"""CLI pour l'analyse forensique intégrée combinant tous les composants du pipeline."""

import argparse
from pathlib import Path

from src.forensics.integrated_pipeline import ForensicPipeline
from src.models import MODEL_REGISTRY, RAW_SIGNAL_MODELS


def main():
    parser = argparse.ArgumentParser(
        description="Integrated forensic analysis of RF signal files"
    )
    # Requis
    parser.add_argument("--file", required=True, help="Path to raw signal CSV file")

    # Configuration du classificateur
    parser.add_argument("--classifier_model", default="resnet",
                        choices=[k for k in MODEL_REGISTRY if k not in RAW_SIGNAL_MODELS],
                        help="Classifier model name (spectrogram-based only — cnn1d not supported)")
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
                        choices=[k for k in MODEL_REGISTRY if k not in RAW_SIGNAL_MODELS],
                        help="Backbone model for Siamese encoder (spectrogram-based only)")
    parser.add_argument("--gallery", default=None,
                        help="Path to .npz gallery of known drone embeddings")

    # Configuration OpenMax
    parser.add_argument("--openmax_params", default=None,
                        help="Path to fitted OpenMax parameters (.pkl)")

    # Configuration GNN
    parser.add_argument("--gnn_weights", default=None,
                        help="Path to GNN weights (.pt) for graph-based multi-segment analysis")
    parser.add_argument("--gnn_emb_dim", type=int, default=128,
                        help="GNN input embedding dimension (must match classifier output dim). "
                             "Defaults by model: smallrf/resnet/transformer/conformer=128, ast=192, efficientnet=1280")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_threshold", type=float, default=0.5,
                        help="Cosine similarity threshold for graph edge creation")
    parser.add_argument("--gnn_k", type=int, default=5,
                        help="k-NN fallback: minimum neighbours per node in the graph")

    # Configuration générale
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--anomaly_threshold", type=float, default=0.7,
                        help="Confidence threshold for classification anomaly flagging")
    parser.add_argument("--explain_segments", default="anomalous",
                        choices=["anomalous", "drone", "all", "none"],
                        help="Which segments to explain with Grad-CAM/attention: "
                             "'anomalous' (low-confidence or OpenMax unknown, default), "
                             "'drone' (all drone detections), "
                             "'all' (every segment), "
                             "'none' (disable)")
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
            f"outputs/dronerf_{args.classifier_model}_{args.task}/models/best_model.pt"
        )

    if args.output_dir is None:
        file_stem = Path(args.file).stem
        args.output_dir = f"outputs/forensic_integrated/{file_stem}"

    # Warn if gnn_emb_dim doesn't match the selected classifier's embedding dim
    _EMB_DIMS = {"ast": 192, "efficientnet": 1280}
    expected_dim = _EMB_DIMS.get(args.classifier_model, 128)
    if args.gnn_weights and args.gnn_emb_dim != expected_dim:
        print(f"WARNING: --gnn_emb_dim={args.gnn_emb_dim} but {args.classifier_model} "
              f"produces {expected_dim}-dim embeddings. "
              f"Pass --gnn_emb_dim {expected_dim} or the GNN will fail.")

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
        "gnn_weights": args.gnn_weights,
        "gnn_emb_dim": args.gnn_emb_dim,
        "gnn_hidden_dim": args.gnn_hidden_dim,
        "gnn_threshold": args.gnn_threshold,
        "gnn_k": args.gnn_k,
        "device": device,
        "anomaly_threshold": args.anomaly_threshold,
        "explain_segments": args.explain_segments,
    }

    # Initialiser le pipeline
    pipeline = ForensicPipeline(config)

    # Analyser le fichier
    print(f"\nAnalyzing: {args.file}")
    timeline = pipeline.analyze_file(args.file, output_dir=args.output_dir)

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

    gnn_summary = report.get("gnn_summary", {})
    if gnn_summary.get("segments_analyzed", 0) > 0:
        print(f"\n  GNN segments:      {gnn_summary['segments_analyzed']}")
        print(f"  GNN drone segs:    {gnn_summary['drone_segments']}")
        print(f"  GNN avg conf:      {gnn_summary['average_confidence']}")
        print(f"  GNN class dist:")
        for cls, count in gnn_summary.get("class_distribution", {}).items():
            print(f"    {cls}: {count}")

    expl_summary = report.get("explainability_summary", {})
    if expl_summary.get("segments_explained", 0) > 0:
        print(f"\n  Explainability ({expl_summary.get('method', '?')}):")
        print(f"    Segments explained: {expl_summary['segments_explained']} "
              f"(mode: {expl_summary.get('explain_mode', '?')})")
        print(f"    Heatmaps: {args.output_dir}/explainability/")

    components = report["report_metadata"]["components"]
    active = [k for k, v in components.items() if v]
    print(f"\n  Active components: {', '.join(active)}")
    print(f"{'='*60}")
    print(f"  Report: {args.output_dir}/integrated_forensic_report.json")
    print(f"  Timeline: {args.output_dir}/integrated_timeline.png")


if __name__ == "__main__":
    main()
