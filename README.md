# RF Drone Forensics

AI-based RF drone forensics pipeline for detection, attribution, open-set recognition, robustness evaluation, explainability, and forensic reporting using spectrograms and deep learning.

**Datasets:** DroneRF (4 classes), CageDroneRF (27 classes), RFUAV (37 drone types)

**Models:** SmallRFNet (CNN), RFResNet, RFTransformer, SVM, Random Forest

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
python -m src.training.train_multimodel --dataset cagedronerf --model resnet --task multiclass --epochs 20
python -m src.training.train_multimodel --dataset rfuav --model transformer --task multiclass --epochs 20

# Traditional ML baselines (SVM + Random Forest)
python -m src.training.train_baselines --dataset all
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

# Multiclass robustness + open-set
python -m src.evaluation.run_multiclass_eval

# Combined model evaluation (robustness + explainability + baselines + open-set)
python -m src.evaluation.run_combined_evaluation

# Cross-dataset generalization (leave-one-out, pairwise, pretrain+fine-tune)
python -m src.evaluation.cross_dataset_enhanced --model resnet --epochs 20

# Generate thesis figures
python -m src.evaluation.plot_baselines_comparison
python -m src.evaluation.plot_cross_dataset_figures
```

### 7. Forensic analysis

```bash
# Single file
python -m src.forensics.run_forensic_analysis --file "data/raw/DroneRF/AR drone/10100H_0.csv"
python -m src.forensics.run_forensic_analysis --file path/to/signal.csv --model resnet --task multiclass

# Batch (entire dataset)
python -m src.forensics.run_forensic_batch --folder "data/raw/DroneRF" --recursive --model resnet --task multiclass
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
│   │   ├── cagedronerf_dataset.py         # CageDroneRF PNG loader (27 classes)
│   │   ├── rfuav_dataset.py               # RFUAV JPG loader (37 classes)
│   │   └── download_rfuav.py              # Download RFUAV from HuggingFace
│   ├── preprocessing/
│   │   ├── segmentation.py                # Sliding window function
│   │   ├── stft_utils.py                  # STFT + log-magnitude normalization
│   │   └── precompute_spectrograms.py     # Batch precompute .npy files
│   ├── models/
│   │   ├── cnn_spectrogram.py             # SmallRFNet (155K params)
│   │   ├── resnet_spectrogram.py          # RFResNet (697K params)
│   │   └── transformer_spectrogram.py     # RFTransformer (375K params)
│   ├── training/
│   │   ├── train_multimodel.py            # Unified training (all models x datasets x tasks)
│   │   ├── train_baselines.py             # SVM + Random Forest
│   │   └── run_all_experiments.py         # Master experiment runner
│   ├── evaluation/
│   │   ├── metrics.py                     # Accuracy, F1, ROC-AUC, ECE, plots
│   │   ├── feature_extraction.py          # Handcrafted features for baselines
│   │   ├── robustness.py                  # SNR noise injection
│   │   ├── openset.py                     # MSP, Energy, Mahalanobis OOD
│   │   ├── explainability.py              # Grad-CAM heatmaps
│   │   ├── cross_dataset_enhanced.py      # Leave-one-out, pairwise, fine-tune
│   │   ├── run_combined_evaluation.py     # Combined model evaluation
│   │   ├── run_multiclass_eval.py         # Multiclass robustness + open-set
│   │   ├── run_binary_robustness.py       # Per-dataset binary robustness
│   │   ├── plot_baselines_comparison.py   # Thesis figures: DL vs baselines
│   │   └── plot_cross_dataset_figures.py  # Thesis figures: cross-dataset
│   └── forensics/
│       ├── timeline.py                    # Per-segment classification + anomaly
│       ├── run_forensic_analysis.py       # Single file forensic report
│       └── run_forensic_batch.py          # Batch forensic analysis
├── data/                    # Not in Git — downloaded/generated locally
├── outputs/                 # Not in Git — generated by training/evaluation
├── requirements.txt
├── CLAUDE.md                # Claude Code instructions
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
| SmallRFNet | CNN | 155K | 3x (Conv+BN+ReLU+MaxPool) -> AdaptiveAvgPool -> FC + Dropout |
| RFResNet | ResNet | 697K | 3 residual blocks (32->64->128 channels) -> AdaptiveAvgPool -> FC |
| RFTransformer | Hybrid CNN-Transformer | 375K | CNN stem (3x stride-2 conv) -> 128 tokens -> 2 Transformer blocks (4-head attention) -> CLS token -> FC |

All models take input shape `[batch, 1, H, W]` (single-channel spectrogram).

## Key Technical Details

- **Segment window:** 131,072 samples, hop 65,536 (50% overlap)
- **STFT:** nperseg=512, noverlap=256, fs=1.0
- **Normalization:** log-magnitude, zero-mean, unit-variance
- **Splits:** 70% train / 15% val / 15% test (file-level stratified, no data leakage)
- **Training:** Adam lr=1e-3, CosineAnnealingLR, CrossEntropyLoss with class weights, SpecAugment, 20 epochs, batch_size=16
- **Open-set methods:** MSP, Energy scoring, Mahalanobis distance
- **Explainability:** Grad-CAM on last convolutional layer
- **Forensic analysis:** Segment-by-segment classification, confidence tracking, anomaly detection (threshold=0.7), structured JSON reports
- **Windows note:** DataLoaders use `num_workers=0` for compatibility
