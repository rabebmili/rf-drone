"""Forensic timeline generation: segment-by-segment classification with confidence and anomaly flags."""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.segmentation import segment_signal
from src.preprocessing.stft_utils import compute_log_spectrogram


def analyze_signal_file(model, file_path, device, class_names=None,
                        window_size=131072, hop_size=65536,
                        fs=1.0, nperseg=512, noverlap=256,
                        anomaly_threshold=0.7):
    """Analyze a signal file and produce a forensic timeline (list of per-segment dicts).

    Args:
        anomaly_threshold: max softmax probability below which a segment is flagged as anomalous.
    """
    if class_names is None:
        class_names = ["Background", "Drone"]

    signal = load_dronerf_csv(file_path)
    segments = segment_signal(signal, window_size=window_size, hop_size=hop_size)

    model.eval()
    timeline = []

    for i, seg in enumerate(segments):
        start_sample = i * hop_size
        end_sample = start_sample + window_size

        # Calculer le spectrogramme
        _, _, S_log = compute_log_spectrogram(seg, fs=fs, nperseg=nperseg, noverlap=noverlap)

        # Préparer le tenseur d'entrée
        x = torch.tensor(S_log, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Passe avant
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

            # Score d'anomalie basé sur l'énergie
            energy = torch.logsumexp(logits, dim=1).item()

        # Déterminer si le segment est suspect
        is_anomalous = confidence < anomaly_threshold

        entry = {
            "segment_id": i,
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "start_time_s": start_sample / fs if fs > 0 else start_sample,
            "end_time_s": end_sample / fs if fs > 0 else end_sample,
            "predicted_class": int(pred),
            "predicted_label": class_names[pred] if pred < len(class_names) else f"Unknown ({pred})",
            "confidence": round(float(confidence), 4),
            "energy_score": round(float(energy), 4),
            "is_anomalous": bool(is_anomalous),
            "all_probabilities": {
                class_names[j] if j < len(class_names) else f"class_{j}": round(float(probs[0, j]), 4)
                for j in range(probs.shape[1])
            },
        }
        timeline.append(entry)

    return timeline


def generate_forensic_report(timeline, file_path, output_path, class_names=None):
    """Generate a structured JSON forensic report from a timeline and save to output_path."""
    if class_names is None:
        class_names = ["Background", "Drone"]

    # Statistiques récapitulatives
    total_segments = len(timeline)
    anomalous_segments = [e for e in timeline if e["is_anomalous"]]
    drone_segments = [e for e in timeline if e["predicted_class"] != 0]

    class_distribution = {}
    for entry in timeline:
        label = entry["predicted_label"]
        class_distribution[label] = class_distribution.get(label, 0) + 1

    avg_confidence = np.mean([e["confidence"] for e in timeline])

    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "source_file": str(file_path),
            "total_segments": total_segments,
            "framework": "RF Drone Forensics Pipeline",
        },
        "summary": {
            "class_distribution": class_distribution,
            "drone_segments_count": len(drone_segments),
            "anomalous_segments_count": len(anomalous_segments),
            "average_confidence": round(float(avg_confidence), 4),
            "anomalous_segment_ids": [e["segment_id"] for e in anomalous_segments],
        },
        "timeline": timeline,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Rapport forensique sauvegardé : {output_path}")
    return report


def plot_forensic_timeline(timeline, output_path=None, title="Forensic Timeline"):
    """Plot the forensic timeline as a multi-panel strip chart."""
    segments = list(range(len(timeline)))
    confidences = [e["confidence"] for e in timeline]
    classes = [e["predicted_class"] for e in timeline]
    anomalous = [e["is_anomalous"] for e in timeline]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Prédictions de classe au fil du temps
    axes[0].bar(segments, classes, color=["red" if c > 0 else "blue" for c in classes], alpha=0.7)
    axes[0].set_ylabel("Predicted Class")
    axes[0].set_title(f"{title} — Classification")

    # Confiance au fil du temps
    axes[1].plot(segments, confidences, "g-", linewidth=1)
    axes[1].fill_between(segments, confidences, alpha=0.3, color="green")
    axes[1].axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="Anomaly threshold")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Confidence Score")
    axes[1].legend()

    # Indicateurs d'anomalie
    anomaly_vals = [1 if a else 0 for a in anomalous]
    axes[2].bar(segments, anomaly_vals, color="red", alpha=0.7)
    axes[2].set_ylabel("Anomaly Flag")
    axes[2].set_xlabel("Segment Index")
    axes[2].set_title("Suspicious Segments")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Graphique de la chronologie sauvegardé : {output_path}")
    plt.close()
