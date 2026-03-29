from pathlib import Path
import pandas as pd
from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.segmentation import segment_boundaries


WINDOW_SIZE = 131072
HOP_SIZE = 65536


def build_segment_index(
    metadata_csv="data/metadata/dronerf_metadata.csv",
    output_csv="data/metadata/dronerf_segments.csv",
    max_files=None
):
    df = pd.read_csv(metadata_csv)
    rows = []

    if max_files is not None:
        df = df.head(max_files)

    for i, row in df.iterrows():
        file_path = row["file_path"]
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{len(df)}] Processing files...")

        signal = load_dronerf_csv(file_path)
        bounds = segment_boundaries(len(signal), WINDOW_SIZE, HOP_SIZE)

        for segment_id, (start, end) in enumerate(bounds):
            rows.append({
                "file_path": file_path,
                "segment_id": segment_id,
                "start": start,
                "end": end,
                "label_binary": row["label_binary"],
                "label_multiclass": row["label_multiclass"],
                "label_class_name": row["label_class_name"],
                "activity_code": row["activity_code"],
                "sample_id": row["sample_id"],
                "dataset": row["dataset"]
            })

    seg_df = pd.DataFrame(rows)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_csv(output_path, index=False)

    print(f"Segment index saved: {output_path} ({len(seg_df)} segments)")


if __name__ == "__main__":
    build_segment_index(max_files=None)