from pathlib import Path
import pandas as pd


CLASS_MAP = {
    "Background RF activites": 0,
    "AR drone": 1,
    "Bepop drone": 2,
    "Phantom drone": 3,
}


def infer_labels_from_path(path: Path):
    parent = path.parent.name
    stem = path.stem  # ex: 10100H_0

    # Étiquette binaire : 0 = fond, 1 = drone
    label_binary = 0 if parent == "Background RF activites" else 1

    # Étiquette multiclasse : 0-3 selon le dossier
    if parent not in CLASS_MAP:
        raise ValueError(f"Classe inconnue détectée : {parent}")
    label_multiclass = CLASS_MAP[parent]

    # Code activité et ID échantillon depuis le nom du fichier (ex: 10100H_0 -> 10100H, 0)
    parts = stem.split("_")
    activity_code = parts[0] if len(parts) > 0 else ""
    sample_id = parts[1] if len(parts) > 1 else ""

    return {
        "file_path": str(path),
        "label_binary": label_binary,
        "label_class_name": parent,
        "label_multiclass": label_multiclass,
        "activity_code": activity_code,
        "sample_id": sample_id,
        "file_stem": stem,
        "dataset": "DroneRF"
    }


def build_metadata(root="data/raw/DroneRF", output_csv="data/metadata/dronerf_metadata.csv"):
    root = Path(root)
    rows = []

    for file in sorted(root.rglob("*.csv")):
        rows.append(infer_labels_from_path(file))

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"Metadata saved: {output_csv} ({len(df)} files)")


if __name__ == "__main__":
    build_metadata()