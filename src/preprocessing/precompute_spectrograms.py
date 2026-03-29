from pathlib import Path
import numpy as np
import pandas as pd

from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.stft_utils import compute_log_spectrogram


def main(
    segments_csv="data/metadata/dronerf_segments_split.csv",
    output_csv="data/metadata/dronerf_precomputed.csv",
    output_dir="data/processed/dronerf_spectrograms",
    fs=1.0,
    nperseg=512,
    noverlap=256
):
    df = pd.read_csv(segments_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    current_file = None
    current_signal = None

    for i, row in df.iterrows():
        file_path = row["file_path"]

        # Éviter de relire le même fichier pour les segments successifs
        if current_file != file_path:
            current_signal = load_dronerf_csv(file_path)
            current_file = file_path
            if (i + 1) % 100 == 0 or i == 0:
                print(f"[{i+1}/{len(df)}] Processing segments...")

        start = int(row["start"])
        end = int(row["end"])
        segment = current_signal[start:end]

        _, _, S_log = compute_log_spectrogram(
            segment,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap
        )

        spec_name = f"spec_{i:06d}.npy"
        spec_path = out_dir / spec_name
        np.save(spec_path, S_log.astype(np.float32))

        rows.append({
            "spec_path": str(spec_path),
            "file_path": row["file_path"],
            "segment_id": row["segment_id"],
            "start": row["start"],
            "end": row["end"],
            "label_binary": row["label_binary"],
            "label_multiclass": row["label_multiclass"],
            "label_class_name": row["label_class_name"],
            "activity_code": row["activity_code"],
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "split": row["split"]
        })

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Precomputation done. CSV: {out_csv}, spectrograms: {out_dir}")


if __name__ == "__main__":
    main()