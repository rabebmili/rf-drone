"""Fit OpenMax parameters and save them as a .pkl file for use in the integrated forensic pipeline.

Usage:
    python -m src.forensics.build_openmax_params \\
        --dataset dronerf --model ast --task multiclass

    python -m src.forensics.build_openmax_params \\
        --dataset cagedronerf --model resnet --task multiclass \\
        --output outputs/openmax_params_cagedronerf_resnet_multiclass.pkl

The output .pkl file can then be passed to run_integrated_analysis.py via:
    --openmax_params outputs/openmax_params_dronerf_ast_multiclass.pkl
"""

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models import MODEL_REGISTRY, RAW_SIGNAL_MODELS, get_model
from src.evaluation.openset import fit_openmax
from src.training.train_multimodel import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Fit and save OpenMax (EVT/Weibull) parameters for the forensic pipeline"
    )
    parser.add_argument("--dataset", default="dronerf",
                        choices=["dronerf", "cagedronerf", "rfuav"],
                        help="Dataset used to train the model")
    parser.add_argument("--model", default="ast",
                        choices=[k for k in MODEL_REGISTRY if k not in RAW_SIGNAL_MODELS],
                        help="Model architecture key (spectrogram-based only — must match forensic pipeline)")
    parser.add_argument("--task", default="multiclass",
                        choices=["binary", "multiclass"])
    parser.add_argument("--weights", default=None,
                        help="Path to model weights .pt (auto-detected if omitted)")
    parser.add_argument("--output", default=None,
                        help="Output .pkl path (auto-named from dataset/model/task if omitted)")
    parser.add_argument("--tail_size", type=int, default=20,
                        help="Weibull tail size for EVT fitting (default: 20)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect weights path
    if args.weights is None:
        args.weights = f"outputs/{args.dataset}_{args.model}_{args.task}/models/best_model.pt"
    if not Path(args.weights).exists():
        print(f"ERROR: weights not found at {args.weights}")
        print("Train the model first or pass --weights explicitly.")
        return

    # Auto-name output path
    if args.output is None:
        args.output = f"outputs/openmax_params_{args.dataset}_{args.model}_{args.task}.pkl"

    # Load dataset (training split only — OpenMax is fitted on train)
    print(f"Loading {args.dataset}/{args.task} training split...")
    train_ds, _, _, num_classes, class_names = load_dataset(
        args.dataset, args.task, model_name=args.model
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Load model
    print(f"Loading {args.model} weights from {args.weights}...")
    model = get_model(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=device))
    model.eval()

    # Fit OpenMax
    print(f"Fitting OpenMax ({num_classes} classes, tail_size={args.tail_size})...")
    mavs, weibull_params = fit_openmax(model, train_loader, device, num_classes,
                                        tail_size=args.tail_size)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    params = {
        "mavs": mavs,
        "weibull_params": weibull_params,
        "num_classes": num_classes,
        "class_names": class_names,
        "dataset": args.dataset,
        "model": args.model,
        "task": args.task,
    }
    with open(args.output, "wb") as f:
        pickle.dump(params, f)

    print(f"\nOpenMax params saved to: {args.output}")
    print(f"  Classes : {class_names}")
    print(f"  MAVs    : {len(mavs)} vectors (dim {mavs[0].shape[0] if hasattr(mavs[0], 'shape') else 'n/a'})")
    print(f"\nPass to integrated analysis with:")
    print(f"  --openmax_params {args.output}")


if __name__ == "__main__":
    main()
