# Pipeline Map — RF Drone Forensics

Complete mapping of every script to its role and outputs.

---

## Data Preparation (run once, in order)

| Script | Role | Outputs |
|--------|------|---------|
| `src/datasets/build_dronerf_metadata.py` | Scans raw CSVs, infers labels | `data/metadata/dronerf_metadata.csv` |
| `src/datasets/build_dronerf_segments.py` | Sliding-window segment index | `data/metadata/dronerf_segments.csv` |
| `src/datasets/split_segments_by_file.py` | File-level train/val/test split | `data/metadata/dronerf_segments_split.csv` |
| `src/preprocessing/precompute_spectrograms.py` | STFT all segments to .npy | `data/processed/dronerf_spectrograms/*.npy` + `data/metadata/dronerf_precomputed.csv` |
| `src/datasets/download_rfuav.py` | Downloads RFUAV from HuggingFace | `data/raw/RFUAV/` |

---

## Libraries (no direct output, called by others)

| Script | What it provides | Used by |
|--------|-----------------|---------|
| `src/datasets/load_signal.py` | `load_dronerf_csv()` — reads raw CSV | build_dronerf_segments, precompute_spectrograms, timeline |
| `src/datasets/dronerf_precomputed_dataset.py` | `DroneRFPrecomputedDataset` class | train_multimodel, run_all_experiments, all evaluation runners |
| `src/datasets/cagedronerf_dataset.py` | `create_cagedronerf_loaders()` | train_multimodel, all evaluation runners, cross_dataset_enhanced |
| `src/datasets/rfuav_dataset.py` | `create_rfuav_loaders()` | train_multimodel, run_multiclass_eval, run_combined_evaluation, cross_dataset_enhanced |
| `src/preprocessing/segmentation.py` | `segment_signal()` | build_dronerf_segments, timeline |
| `src/preprocessing/stft_utils.py` | `compute_log_spectrogram()` | precompute_spectrograms, timeline |
| `src/models/cnn_spectrogram.py` | `SmallRFNet` architecture | train_multimodel, run_all_experiments, forensics, cross_dataset_enhanced |
| `src/models/resnet_spectrogram.py` | `RFResNet` architecture | all training/evaluation/forensic scripts |
| `src/models/transformer_spectrogram.py` | `RFTransformer` architecture | train_multimodel, run_all_experiments, forensics, cross_dataset_enhanced |
| `src/evaluation/metrics.py` | `full_evaluation()` — CM, ROC, PR, calibration | train_multimodel, run_all_experiments, cross_dataset_enhanced |
| `src/evaluation/feature_extraction.py` | Handcrafted spectrogram features | train_baselines, run_combined_evaluation |
| `src/evaluation/robustness.py` | `run_robustness_evaluation()` | run_all_experiments, run_binary_robustness, run_multiclass_eval, run_combined_evaluation |
| `src/evaluation/openset.py` | MSP/Energy/Mahalanobis OOD detection | run_all_experiments, run_multiclass_eval, run_combined_evaluation |
| `src/evaluation/explainability.py` | Grad-CAM heatmaps | run_all_experiments, run_combined_evaluation |
| `src/forensics/timeline.py` | `analyze_signal_file()`, report/plot generation | run_forensic_analysis, run_forensic_batch |

---

## Training (produces models + figures)

| Script | Output dir | Outputs |
|--------|-----------|---------|
| `src/training/train_multimodel.py` | `outputs/{dataset}_{model}_{task}/` | `figures/training_curves.png`, `figures/confusion_matrix.png`, `figures/confusion_matrix_normalized.png`, `figures/roc_curves.png`, `figures/pr_curves.png`, `figures/calibration_diagram.png`, `models/best_model.pt`, `results.json` |
| `src/training/train_baselines.py` | `outputs/baselines_{dataset}_{task}/` | `svm_model.joblib`, `rf_model.joblib`, `*_scaler.joblib`, `svm_results.json`, `random_forest_results.json` |
| `src/training/run_all_experiments.py` | Orchestrates all above | Calls train_multimodel + robustness + openset + explainability sequentially |

---

## Evaluation (produces analysis figures)

| Script | Output dir | Figures generated |
|--------|-----------|-------------------|
| `src/evaluation/run_binary_robustness.py` | `outputs/robustness_single_dataset_binary/` | `{DroneRF,CageDroneRF}/robustness_vs_snr.png` — per-dataset binary model robustness |
| `src/evaluation/run_multiclass_eval.py` | `outputs/multiclass_evaluation/` | `robustness_multiclass/{dataset}/robustness_vs_snr.png` + `openset_multiclass/{dataset}/holdout_{class}/energy_distribution.png`, `msp_distribution.png` |
| `src/evaluation/run_combined_evaluation.py` | `outputs/evaluation_combined_model/` | `robustness/{dataset}/robustness_vs_snr.png` + `explainability/{dataset}/gradcam_*.png` + `openset/holdout_{class}/energy_distribution.png`, `msp_distribution.png` |
| `src/evaluation/cross_dataset_enhanced.py` | `outputs/cross_dataset_enhanced/` | `{experiment}/*_confusion_matrix.png` + `*_metrics.json` + `models/*_best.pt` |
| `src/evaluation/plot_baselines_comparison.py` | `outputs/thesis_figures/` | `baselines_binary_comparison.png`, `baselines_multiclass_comparison.png`, `baselines_dl_advantage.png` |
| `src/evaluation/plot_cross_dataset_figures.py` | `outputs/thesis_figures/` | `cross_dataset_heatmap.png`, `leave_one_out_bar.png`, `finetune_comparison.png`, `domain_shift_summary.png`, `ablation_comparison.png`, `model_comparison_all_datasets.png` |

---

## Forensics (produces investigation reports)

| Script | Output dir | Outputs |
|--------|-----------|---------|
| `src/forensics/run_forensic_analysis.py` | `outputs/forensic_reports/{file_stem}/` | `forensic_report.json` + `forensic_timeline.png` |
| `src/forensics/run_forensic_batch.py` | `outputs/forensic_batch/{model}_{task}/` | Per-file: `*_report.json` + `*_timeline.png`. Global: `global_forensic_report.json` + `global_class_distribution.png` + `global_confidence_distribution.png` + `global_per_file_confidence.png` + `global_detection_rate_by_folder.png` |

---

## Pipeline Flow

```
[Data Prep]
  build_dronerf_metadata --> build_dronerf_segments --> split_segments_by_file --> precompute_spectrograms
  download_rfuav

[Training]
  train_multimodel ----> outputs/{dataset}_{model}_{task}/  (models + 6 figures)
  train_baselines  ----> outputs/baselines_{dataset}_{task}/ (joblib + json)

[Evaluation]
  run_binary_robustness      ----> outputs/robustness_single_dataset_binary/
  run_multiclass_eval        ----> outputs/multiclass_evaluation/
  run_combined_evaluation    ----> outputs/evaluation_combined_model/
  cross_dataset_enhanced     ----> outputs/cross_dataset_enhanced/
  plot_baselines_comparison  ----> outputs/thesis_figures/
  plot_cross_dataset_figures ----> outputs/thesis_figures/

[Forensics]
  run_forensic_analysis ----> outputs/forensic_reports/
  run_forensic_batch    ----> outputs/forensic_batch/
```
