# Pipeline Map

Detailed script-to-output mapping for all 9 pipeline phases.

---

## 1. Data Preparation
| Script | Input | Output |
|--------|-------|--------|
| `src/datasets/build_dronerf_metadata.py` | `data/raw/DroneRF/` folder hierarchy | `data/metadata/dronerf_metadata.csv` |
| `src/datasets/build_dronerf_segments.py` | `data/metadata/dronerf_metadata.csv` | `data/metadata/dronerf_segments.csv` |
| `src/datasets/split_segments_by_file.py` | `data/metadata/dronerf_segments.csv` | `data/metadata/dronerf_segments_split.csv` |
| `src/preprocessing/precompute_spectrograms.py` | `data/metadata/dronerf_segments_split.csv` + raw CSVs | `data/processed/dronerf_spectrograms/*.npy` |
| `src/datasets/download_rfuav.py` | HuggingFace remote | `data/raw/RFUAV/*.jpg` (5,679 images, 37 classes) |

## 2. Libraries (Models, Datasets, Utilities)

| File | Exports | Used by |
|------|---------|---------|
| `src/models/__init__.py` | `MODEL_REGISTRY`, `get_model()` | all training/evaluation scripts |
| `src/models/cnn_spectrogram.py` | `SmallRFNet` | train_multimodel |
| `src/models/resnet_spectrogram.py` | `RFResNet` | train_multimodel |
| `src/models/transformer_spectrogram.py` | `RFTransformer` | train_multimodel |
| `src/models/efficientnet_spectrogram.py` | `RFEfficientNet` | train_multimodel |
| `src/models/ast_spectrogram.py` | `RFAST` | train_multimodel |
| `src/models/conformer_spectrogram.py` | `RFConformer` | train_multimodel |
| `src/models/cnn_1d.py` | `RFCNN1D` | train_multimodel |
| `src/models/siamese_network.py` | `SiameseNetwork` | train_siamese, build_gallery, train_gnn |
| `src/models/vae.py` | `RFVAE` | train_vae, integrated_pipeline |
| `src/models/gnn.py` | `RFDroneGNN`, `build_similarity_graph()` | train_gnn, integrated_pipeline |
| `src/models/ensemble.py` | `EnsembleCNNTransformer` | eval_ensemble |
| `src/datasets/dronerf_precomputed_dataset.py` | `DroneRFPrecomputedDataset` | train_multimodel |
| `src/datasets/dronerf_raw_dataset.py` | `DroneRFRawDataset` | train_multimodel (cnn1d) |
| `src/datasets/cagedronerf_dataset.py` | `CageDroneRFDataset` | train_multimodel |
| `src/datasets/rfuav_dataset.py` | `RFUAVDataset` | train_multimodel |
| `src/datasets/siamese_dataset.py` | `TripletDataset` | train_siamese |
| `src/datasets/signal_graph_dataset.py` | `SignalGraphDataset` | train_gnn |
| `src/preprocessing/segmentation.py` | `segment_signal()` | precompute, forensics |
| `src/preprocessing/stft_utils.py` | `compute_log_spectrogram()` | precompute, forensics |
| `src/evaluation/metrics.py` | accuracy, F1, ROC-AUC, ECE, confusion matrix, PR curves | all evaluation scripts |
| `src/evaluation/feature_extraction.py` | handcrafted spectrogram features | train_baselines |
| `src/evaluation/robustness.py` | SNR noise injection | run_binary_robustness, run_multiclass_eval |
| `src/evaluation/openset.py` | MSP, Energy, Mahalanobis, OpenMax (EVT/Weibull) | run_multiclass_eval, integrated_pipeline |
| `src/evaluation/explainability.py` | Grad-CAM, Attention Rollout, GradCAM1D | run_combined_evaluation, integrated_pipeline |
| `src/forensics/timeline.py` | segment-by-segment classification + anomaly detection | run_forensic_analysis, run_forensic_batch |
| `src/forensics/integrated_pipeline.py` | `ForensicPipeline` class | run_integrated_analysis |

## 3. Training (Classification)

| Script | Output directory | Key outputs |
|--------|-----------------|-------------|
| `src/training/train_multimodel.py` | `outputs/{dataset}_{model}_{task}/` | `models/best_model.pt`, `figures/training_curves.png`, `figures/confusion_matrix.png`, `results.json` |
| `src/training/train_baselines.py` | `outputs/baselines_{dataset}_{task}/` | `models/svm.pkl`, `models/rf.pkl`, `results.json` |

## 4. Training (Siamese / VAE / GNN)

| Script | Output directory | Key outputs |
|--------|-----------------|-------------|
| `src/training/train_siamese.py` | `outputs/siamese_{dataset}_{backbone}_{task}/` | `models/best_siamese.pt`, `results.json` |
| `src/training/train_vae.py` | `outputs/vae_{dataset}/` | `models/best_vae.pt`, `figures/vae_training_curves.png`, `figures/vae_reconstructions.png`, `results.json` |
| `src/training/train_gnn.py` | `outputs/gnn_{dataset}_{task}/` | `models/best_gnn.pt`, `figures/gnn_training_curves.png`, `results.json` |

## 5. Training (Master Runner)

| Script | Runs |
|--------|------|
| `src/training/run_all_experiments.py` | All models x datasets x tasks, baselines, robustness, open-set, explainability |

## 6. Evaluation (Robustness + Open-Set)

| Script | Output directory | Key outputs |
|--------|-----------------|-------------|
| `src/evaluation/run_binary_robustness.py` | `outputs/{dataset}_{model}_binary/` | robustness curves, SNR results |
| `src/evaluation/run_multiclass_eval.py` | `outputs/{dataset}_{model}_multiclass/` | robustness + open-set (MSP, Energy, Mahalanobis, OpenMax) |
| `src/evaluation/run_combined_evaluation.py` | `outputs/combined_evaluation/` | robustness + explainability + baselines across all 3 datasets |
| `src/evaluation/eval_ensemble.py` | `outputs/ensemble_evaluation/` | `ensemble_results.json` |

## 7. Evaluation (Cross-Dataset + Figures)

| Script | Output directory | Key outputs |
|--------|-----------------|-------------|
| `src/evaluation/cross_dataset_enhanced.py` | `outputs/cross_dataset/` | leave-one-out, pairwise, pretrain+fine-tune results |
| `src/evaluation/plot_baselines_comparison.py` | `outputs/figures/` | thesis figures: DL vs baselines |
| `src/evaluation/plot_cross_dataset_figures.py` | `outputs/figures/` | thesis figures: cross-dataset generalization |

## 8. Forensic Analysis (Basic)

| Script | Output directory | Key outputs |
|--------|-----------------|-------------|
| `src/forensics/run_forensic_analysis.py` | `outputs/forensic_analysis/` | `forensic_report.json`, `timeline.png` |
| `src/forensics/run_forensic_batch.py` | `outputs/forensic_batch/` | per-file reports + `batch_summary.json` |

## 9. Forensic Analysis (Integrated Pipeline)

| Script | Output directory | Key outputs |
|--------|-----------------|-------------|
| `src/forensics/build_gallery.py` | `outputs/` | `gallery_{dataset}_{task}.npz` |
| `src/forensics/run_integrated_analysis.py` | `outputs/forensic_integrated/` | `integrated_forensic_report.json`, `integrated_timeline.png` |

---

## Pipeline Flow

```
1. Data Prep          build_metadata -> build_segments -> split -> precompute_spectrograms
                                                                   download_rfuav (RFUAV)
        |
        v
2. Training           train_multimodel  (7 models x 3 datasets x binary/multiclass)
                      train_baselines   (SVM + RF)
                      train_siamese     (triplet loss, similarity attribution)
                      train_vae         (reconstruction + KL, anomaly detection)
                      train_gnn         (GAT on similarity graph)
        |
        v
3. Evaluation         run_binary_robustness      (SNR robustness per dataset)
                      run_multiclass_eval        (robustness + open-set + OpenMax)
                      run_combined_evaluation    (robustness + explainability + baselines)
                      eval_ensemble              (CNN + Transformer fusion)
                      cross_dataset_enhanced     (leave-one-out, pairwise, fine-tune)
                      plot_baselines_comparison  (thesis figures)
                      plot_cross_dataset_figures (thesis figures)
        |
        v
4. Forensics          build_gallery              (Siamese embeddings for attribution)
                      run_forensic_analysis      (single file, basic)
                      run_forensic_batch         (batch, basic)
                      run_integrated_analysis    (classification + open-set + VAE + Siamese + GNN + Grad-CAM)
```
