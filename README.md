# RF Drone Forensics

An Explainable Open-Set RF Drone Forensics Framework combining spectrogram transformers, similarity learning, graph neural networks, and anomaly detection. PyTorch-based pipeline for classifying drone RF signals using spectrograms and deep learning. Supports binary classification (drone vs. background) and multi-class classification across three datasets: DroneRF (4 classes), CageDroneRF (27 classes), and RFUAV (37 classes). Includes an integrated forensic analysis framework combining classification, open-set detection, anomaly detection, similarity-based attribution, and explainability.

**Datasets:** DroneRF (4 classes), CageDroneRF (27 classes), RFUAV (37 drone types)

**Models:** SmallRFNet, RFResNet, RFTransformer, RFEfficientNet, RFAST (AST), RFConformer, RFCNN1D, SiameseNetwork, RFVAE, RFDroneGNN, EnsembleCNNTransformer, SVM + Random Forest

> For a detailed mapping of every script to its outputs, see [PIPELINE_MAP.md](PIPELINE_MAP.md).

---

## Quick Start

### 1. Clone and install

```bash
git clone git@github.com:rabebmili/rf-drone-forensics.git
cd rf-drone-forensics

python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

### 2. Download datasets

#### DroneRF (required, ~8 GB)

Download from [IEEE DataPort](https://ieee-dataport.org/open-access/drone-remote-controller-rf-signal-dataset) and place files as:

```
data/raw/DroneRF/
  ├── AR drone/                    # 18 CSV files
  ├── Background RF activites/     # 20 CSV files (typo is intentional)
  ├── Bepop drone/                 # 21 CSV files
  └── Phantom drone/               # 21 CSV files
```

#### CageDroneRF (optional, for cross-dataset experiments)

Download the balanced version and place at `data/raw/CageDroneRF/balanced/`.

#### RFUAV (optional, ~5 GB)

```bash
python -m src.datasets.download_rfuav
```

Downloads spectrogram images (5,679 JPGs, 37 drone types) from [HuggingFace](https://huggingface.co/datasets/kitofrank/RFUAV) to `data/raw/RFUAV/`.

### 3. Data preparation (DroneRF, run in order)

```bash
python -m src.datasets.build_dronerf_metadata
python -m src.datasets.build_dronerf_segments
python -m src.datasets.split_segments_by_file
python -m src.preprocessing.precompute_spectrograms
```

### 4. Train models

```bash
# Deep learning (all models x datasets x tasks)
python -m src.training.train_multimodel --dataset dronerf --model resnet --task binary --epochs 20
python -m src.training.train_multimodel --dataset dronerf --model efficientnet --task multiclass --epochs 20
python -m src.training.train_multimodel --dataset dronerf --model ast --task multiclass --epochs 20
python -m src.training.train_multimodel --dataset dronerf --model conformer --task multiclass --epochs 20
python -m src.training.train_multimodel --dataset dronerf --model cnn1d --task multiclass --epochs 20
python -m src.training.train_multimodel --dataset cagedronerf --model resnet --task multiclass --epochs 20
python -m src.training.train_multimodel --dataset rfuav --model transformer --task multiclass --epochs 20

# Traditional ML baselines (SVM + Random Forest)
python -m src.training.train_baselines --dataset all

# Siamese network (triplet loss for similarity-based attribution)
python -m src.training.train_siamese --dataset dronerf --backbone resnet --task multiclass --epochs 20

# VAE (anomaly detection via reconstruction error)
python -m src.training.train_vae --dataset dronerf --epochs 30

# GNN (graph attention network on similarity graph)
python -m src.training.train_gnn --dataset dronerf --task multiclass --backbone resnet \
    --siamese_weights outputs/siamese_dronerf_resnet_multiclass/models/best_siamese.pt --epochs 30

# Build Siamese gallery (known-drone embeddings for attribution)
python -m src.forensics.build_gallery --dataset dronerf --task multiclass \
    --siamese_weights outputs/siamese_dronerf_resnet_multiclass/models/best_siamese.pt
```

### 5. Run all experiments

```bash
python -m src.training.run_all_experiments --task binary
python -m src.training.run_all_experiments --task multiclass
```

### 6. Evaluation

```bash
# Binary robustness (per-dataset models)
python -m src.evaluation.run_binary_robustness

# Multiclass robustness + open-set (includes OpenMax/EVT)
python -m src.evaluation.run_multiclass_eval

# Combined model evaluation (robustness + explainability + baselines + open-set)
python -m src.evaluation.run_combined_evaluation

# Cross-dataset generalization (leave-one-out, pairwise, pretrain+fine-tune)
python -m src.evaluation.cross_dataset_enhanced --model resnet --epochs 20

# Ensemble evaluation (CNN + Transformer fusion)
python -m src.evaluation.eval_ensemble

# Full open-set pipeline (OpenMax + distribution plots + AUROC summary + thesis figures)
python -m src.evaluation.run_openset_pipeline

# Generate thesis figures
python -m src.evaluation.plot_baselines_comparison
python -m src.evaluation.plot_cross_dataset_figures
```

### 7. Forensic analysis

```bash
# Single file (basic -- classification only)
python -m src.forensics.run_forensic_analysis --file "data/raw/DroneRF/AR drone/10100H_0.csv"
python -m src.forensics.run_forensic_analysis --file path/to/signal.csv --model resnet --task multiclass

# Batch (entire dataset)
python -m src.forensics.run_forensic_batch --folder "data/raw/DroneRF" --recursive --model resnet --task multiclass

# Build prerequisites for integrated pipeline
python -m src.forensics.build_gallery --dataset dronerf --task multiclass \
    --siamese_weights outputs/siamese_dronerf_resnet_multiclass/models/best_siamese.pt
python -m src.forensics.build_openmax_params --dataset dronerf --model ast --task multiclass

# Integrated forensic analysis (classification + open-set + VAE anomaly + Siamese attribution + GNN + Grad-CAM)
python -m src.forensics.run_integrated_analysis \
    --file "data/raw/DroneRF/AR drone/10100H_0.csv" \
    --classifier_model ast --task multiclass \
    --vae_weights outputs/vae_dronerf/models/best_vae.pt \
    --siamese_weights outputs/siamese_dronerf_resnet_multiclass/models/best_siamese.pt \
    --gallery outputs/gallery_dronerf_multiclass.npz \
    --gnn_weights outputs/gnn_dronerf_multiclass/models/best_gnn.pt \
    --openmax_params outputs/openmax_params_dronerf_ast_multiclass.pkl \
    --explain_segments anomalous
```

---

## Project Structure

```
rf-drone-forensics/
├── src/
│   ├── datasets/
│   │   ├── load_signal.py                 # Read raw CSV signals
│   │   ├── build_dronerf_metadata.py      # Scan folders -> metadata CSV
│   │   ├── build_dronerf_segments.py      # Sliding window segment index
│   │   ├── split_segments_by_file.py      # File-level stratified split
│   │   ├── dronerf_precomputed_dataset.py # Precomputed .npy dataset
│   │   ├── dronerf_raw_dataset.py         # Raw signal dataset for 1D-CNN
│   │   ├── cagedronerf_dataset.py         # CageDroneRF PNG loader (27 classes)
│   │   ├── rfuav_dataset.py               # RFUAV JPG loader (37 classes)
│   │   ├── siamese_dataset.py             # TripletDataset for Siamese training
│   │   ├── signal_graph_dataset.py        # SignalGraphDataset for GNN training
│   │   └── download_rfuav.py              # Download RFUAV from HuggingFace
│   ├── preprocessing/
│   │   ├── segmentation.py                # Sliding window function
│   │   ├── stft_utils.py                  # STFT + log-magnitude normalization
│   │   └── precompute_spectrograms.py     # Batch precompute .npy files
│   ├── models/
│   │   ├── __init__.py                    # MODEL_REGISTRY + get_model()
│   │   ├── cnn_spectrogram.py             # SmallRFNet (155K params)
│   │   ├── resnet_spectrogram.py          # RFResNet (696K params)
│   │   ├── transformer_spectrogram.py     # RFTransformer (375K params)
│   │   ├── efficientnet_spectrogram.py    # RFEfficientNet (4.0M params)
│   │   ├── ast_spectrogram.py             # RFAST - Audio Spectrogram Transformer (1.9M params)
│   │   ├── conformer_spectrogram.py       # RFConformer (1.6M params)
│   │   ├── cnn_1d.py                      # RFCNN1D for raw I/Q (183K params)
│   │   ├── siamese_network.py             # SiameseNetwork + triplet loss (~824K params)
│   │   ├── vae.py                         # RFVAE - Conv VAE for anomaly detection (394K params)
│   │   ├── gnn.py                         # RFDroneGNN - GAT on similarity graph (200K params)
│   │   └── ensemble.py                    # EnsembleCNNTransformer (avg/weighted/stacking)
│   ├── training/
│   │   ├── train_multimodel.py            # Unified training (all models x datasets x tasks)
│   │   ├── train_baselines.py             # SVM + Random Forest
│   │   ├── train_siamese.py               # Siamese network (triplet loss)
│   │   ├── train_vae.py                   # VAE (reconstruction + KL divergence)
│   │   ├── train_gnn.py                   # GNN (graph attention network)
│   │   └── run_all_experiments.py         # Master experiment runner
│   ├── evaluation/
│   │   ├── metrics.py                         # Accuracy, F1, ROC-AUC, ECE, plots
│   │   ├── feature_extraction.py              # Handcrafted features for baselines
│   │   ├── robustness.py                      # SNR noise injection
│   │   ├── openset.py                         # MSP, Energy, Mahalanobis, OpenMax (EVT/Weibull)
│   │   ├── explainability.py                  # Grad-CAM, Attention Rollout, GradCAM1D
│   │   ├── cross_dataset_enhanced.py          # Leave-one-out, pairwise, fine-tune
│   │   ├── eval_ensemble.py                   # Ensemble evaluation (CNN + Transformer fusion)
│   │   ├── run_combined_evaluation.py         # Combined model evaluation
│   │   ├── run_multiclass_eval.py             # Multiclass robustness + open-set
│   │   ├── run_binary_robustness.py           # Per-dataset binary robustness
│   │   ├── run_openmax_only.py                # OpenMax for all models × datasets × holdouts
│   │   ├── generate_openmax_plots.py          # OpenMax distribution plots (thesis cases)
│   │   ├── summarize_openset_with_openmax.py  # AUROC summary tables to stdout
│   │   ├── run_openset_pipeline.py            # Orchestrates full open-set pipeline (4 steps)
│   │   ├── plot_openset_thesis_figures.py     # Thesis figures: score distributions, AUROC
│   │   ├── plot_baselines_comparison.py       # Thesis figures: DL vs baselines
│   │   └── plot_cross_dataset_figures.py      # Thesis figures: cross-dataset
│   └── forensics/
│       ├── timeline.py                    # Per-segment classification + anomaly
│       ├── run_forensic_analysis.py       # Single file forensic report
│       ├── run_forensic_batch.py          # Batch forensic analysis
│       ├── integrated_pipeline.py         # ForensicPipeline (classification + open-set + VAE + Siamese + GNN + Grad-CAM)
│       ├── run_integrated_analysis.py     # CLI for integrated forensic analysis
│       ├── build_gallery.py               # Build Siamese embedding gallery
│       └── build_openmax_params.py        # Fit EVT/Weibull → .pkl for integrated pipeline
├── data/                    # Not in Git -- downloaded/generated locally
├── outputs/                 # Not in Git -- generated by training/evaluation
├── requirements.txt
├── PIPELINE_MAP.md          # Detailed script-to-output mapping
└── README.md
```

## Datasets

### DroneRF

| label_multiclass | label_binary | Class Name |
|---|---|---|
| 0 | 0 | Background RF activities |
| 1 | 1 | AR drone |
| 2 | 1 | Bepop drone |
| 3 | 1 | Phantom drone |

- 80 raw CSV files, 4 classes
- Source: [IEEE DataPort](https://ieee-dataport.org/open-access/drone-remote-controller-rf-signal-dataset)

### CageDroneRF

- PNG spectrogram images, 26 drone/device classes + 1 background
- Balanced version used for training
- Source: [Google Drive](https://drive.google.com/drive/folders/1X2A8v8cIb8jPZCfXcXB8rqh8nHdBnXNz)

### RFUAV

- 5,679 JPG spectrogram images, 37 drone/controller types
- All samples are drones (no background class)
- Source: [HuggingFace](https://huggingface.co/datasets/kitofrank/RFUAV)

## Model Architectures

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| SmallRFNet | CNN | 155K | 3-layer CNN baseline (Conv+BN+ReLU+MaxPool) |
| RFResNet | ResNet | 696K | Lightweight residual blocks (32->64->128 channels) |
| RFTransformer | CNN-Transformer | 375K | CNN stem + ViT (2 Transformer blocks, 4-head attention) |
| RFEfficientNet | Transfer Learning | 4.0M | EfficientNet-B0, ImageNet pretrained, adapted for 1-channel |
| RFAST (AST) | Pure Transformer | 1.9M | Patch tokenization 16x16, 4 Transformer layers |
| RFConformer | Conformer | 1.6M | Interleaved Conv + Transformer blocks |
| RFCNN1D | 1D-CNN | 183K | Raw I/Q signal processing, no STFT required |
| SiameseNetwork | Metric Learning | ~824K | Shared encoder + projection head, trained with triplet loss |
| RFVAE | Autoencoder | 394K | Convolutional VAE for unsupervised anomaly detection |
| RFDroneGNN | Graph Neural Net | 200K | 2-layer GAT on similarity graph |
| EnsembleCNNTransformer | Fusion | -- | Average/weighted/stacking of CNN + Transformer |
| SVM + Random Forest | Classical ML | -- | Handcrafted spectrogram features (bandwidth, peak freq, spectral entropy) |

All spectrogram models take input shape `[batch, 1, H, W]` (single-channel spectrogram). RFCNN1D takes `[batch, 1, L]` (raw I/Q signal).

## Key Technical Details

- **Segment window:** 131,072 samples, hop 65,536 (50% overlap)
- **STFT:** nperseg=512, noverlap=256, fs=1.0
- **Normalization:** log-magnitude, zero-mean, unit-variance
- **Splits:** 70% train / 15% val / 15% test (file-level stratified, no data leakage)
- **Training:** Adam lr=1e-3, CosineAnnealingLR, CrossEntropyLoss with class weights, SpecAugment, 20 epochs, batch_size=16
- **Open-set methods:** MSP, Energy scoring, Mahalanobis distance, **OpenMax (EVT/Weibull fitting)**
- **Explainability:** Grad-CAM (CNN layers), **Attention Rollout** (AST/Transformer), **GradCAM1D** (1D-CNN)
- **Forensic analysis:** **Integrated pipeline** with 9 components (classification, open-set detection, VAE anomaly detection, Siamese attribution, GNN graph analysis, Grad-CAM explainability, timeline generation, confidence tracking, report generation) with graceful degradation when optional models are unavailable
- **Siamese attribution:** Triplet loss training, cosine similarity gallery matching for drone type identification
- **GNN:** Graph Attention Network built on Siamese embedding similarity graph for relational analysis
- **Windows note:** DataLoaders use `num_workers=0` for compatibility
