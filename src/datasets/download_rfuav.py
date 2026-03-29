"""Download RFUAV spectrogram images from Hugging Face (requires huggingface_hub)."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download RFUAV spectrogram images")
    parser.add_argument("--output_dir", default="data/raw/RFUAV",
                        help="Where to save the dataset")
    parser.add_argument("--subset", default="spectrograms",
                        choices=["spectrograms", "validation"],
                        help="Which subset to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download, hf_hub_download

    repo_id = "kitofrank/RFUAV"

    if args.subset == "spectrograms":
        print(f"Downloading RFUAV spectrograms to {output_dir}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(output_dir),
                allow_patterns=["ImageSet-AllDrones-MatlabPipeline/**"],
            )
            print(f"Download complete: {output_dir / 'ImageSet-AllDrones-MatlabPipeline'}")
        except Exception as e:
            print(f"Download error: {e}")

    elif args.subset == "validation":
        print(f"Downloading RFUAV validation set to {output_dir}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(output_dir),
                allow_patterns=["ValidationSet_5Drones/**"],
            )
            print(f"Download complete: {output_dir / 'ValidationSet_5Drones'}")
        except Exception as e:
            print(f"Download error: {e}")


if __name__ == "__main__":
    main()
