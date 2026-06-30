"""
Download MedQA-USMLE and PubMedQA datasets from HuggingFace.
Saves raw splits to $HOME/W/data/raw/ on the cluster.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(os.environ.get("HOME", ".")) / "W" / "data" / "raw"


def download_medqa():
    print("Downloading MedQA-USMLE-4-options...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

    out_dir = DATA_DIR / "medqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in dataset:
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for example in tqdm(dataset[split], desc=f"  {split}"):
                f.write(json.dumps(example) + "\n")
        print(f"  Saved {len(dataset[split])} examples → {out_path}")

    return dataset


def download_pubmedqa():
    print("Downloading PubMedQA (pqa_labeled)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

    out_dir = DATA_DIR / "pubmedqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in dataset:
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for example in tqdm(dataset[split], desc=f"  {split}"):
                f.write(json.dumps(example) + "\n")
        print(f"  Saved {len(dataset[split])} examples → {out_path}")

    return dataset


if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    medqa = download_medqa()
    pubmedqa = download_pubmedqa()

    print("\nDownload complete.")
    print(f"MedQA splits: { {k: len(v) for k, v in medqa.items()} }")
    print(f"PubMedQA splits: { {k: len(v) for k, v in pubmedqa.items()} }")
