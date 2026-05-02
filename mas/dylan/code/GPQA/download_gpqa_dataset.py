#!/usr/bin/env python3
"""
Download and prepare the GPQA Diamond dataset for DyLAN experiments.

This mirrors the AIME downloader layout by writing one JSON file per question
into a dataset-specific directory so the DyLAN runners can consume it easily.
"""

import json
import os

from datasets import load_dataset


DATASET_NAME = "fingertap/GPQA-Diamond"
BASE_DIR = "/common/home/vv382/Datasets/GPQA"
OUTPUT_SUBDIR = "diamond"


def _pick_split(dataset):
    """Return the most likely split containing the GPQA Diamond rows."""
    preferred_splits = ["train", "validation", "test"]
    for split_name in preferred_splits:
        if split_name in dataset:
            return dataset[split_name]

    first_split_name = next(iter(dataset.keys()))
    return dataset[first_split_name]


def _normalize_row(row):
    """Convert a GPQA row into the simple DyLAN JSON format."""
    return {
        "question": str(row["question"]),
        "answer": str(row["answer"]),
    }


def download_gpqa_dataset() -> str:
    """Download GPQA Diamond from Hugging Face and store it as per-question JSON files."""
    base_dir = BASE_DIR
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 60)
    print("GPQA Diamond Dataset Download")
    print("=" * 60)
    print()

    print(f"Downloading dataset from Hugging Face: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    print(f"Available splits: {list(ds.keys())}")
    split_data = _pick_split(ds)
    total_problems = len(split_data)
    print(f"Total problems: {total_problems}")
    print()

    output_dir = os.path.join(base_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in enumerate(split_data):
        problem_data = _normalize_row(row)
        filepath = os.path.join(output_dir, f"{idx}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(problem_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved Problem {idx + 1}/{total_problems} → {filepath}")

    print()
    print(f"✓ GPQA Diamond dataset downloaded to {output_dir}")
    print(f"  Total: {total_problems} problems")
    print()
    return output_dir


if __name__ == "__main__":
    output_dir = download_gpqa_dataset()

    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("Dataset structure:")
    print(f"  {output_dir}/")
    print("    0.json  → Problem 1")
    print("    1.json  → Problem 2")
    print("    ...")
    print("    N.json  → Problem N+1")
    print()
