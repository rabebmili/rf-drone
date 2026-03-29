from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main(
    segments_csv="data/metadata/dronerf_segments.csv",
    output_csv="data/metadata/dronerf_segments_split.csv",
    test_size=0.15,
    val_size=0.15,
    random_state=42
):
    df = pd.read_csv(segments_csv)

    # Liste unique des fichiers (split au niveau fichier pour éviter les fuites)
    file_df = df[["file_path", "label_binary", "label_multiclass", "label_class_name"]].drop_duplicates()

    # Split stratifié train+val / test au niveau fichier
    train_val_files, test_files = train_test_split(
        file_df,
        test_size=test_size,
        random_state=random_state,
        stratify=file_df["label_class_name"]
    )

    # Ajuster le ratio val par rapport au sous-ensemble train+val (15% du total ≈ 17.6% de 85%)
    val_ratio_adjusted = val_size / (1.0 - test_size)

    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_files["label_class_name"]
    )

    train_set = set(train_files["file_path"])
    val_set = set(val_files["file_path"])
    test_set = set(test_files["file_path"])

    def assign_split(file_path):
        if file_path in train_set:
            return "train"
        if file_path in val_set:
            return "val"
        if file_path in test_set:
            return "test"
        raise ValueError(f"Fichier non assigné : {file_path}")

    df["split"] = df["file_path"].apply(assign_split)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Split file saved: {output_path} ({len(df)} segments)")


if __name__ == "__main__":
    main()
