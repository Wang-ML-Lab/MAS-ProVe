#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2025-11-19
# @Author  : all
# @Desc    : Split GAIA, AIME24, and AIME25 datasets into validation and test sets

import json
import os
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def split_and_save_dataset(dataset_name: str, hf_dataset_path: str = None, 
                           local_json_path: str = None, validation_size: float = 0.2,
                           output_dir: str = "data", use_random_split: bool = True):
    """
    Load a dataset and split it into validation and test sets.
    
    Args:
        dataset_name: Name for output files (e.g., 'gaia', 'aime24')
        hf_dataset_path: HuggingFace dataset path (e.g., 'simplescaling/aime24_nofigures')
        local_json_path: Local JSON file path (e.g., 'data/datasets/GAIA.json')
        validation_size: Fraction of data for validation (default 0.2 = 20%)
        output_dir: Directory to save output files
        use_random_split: If True, use random split; if False, use first 6 for validation
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    if hf_dataset_path:
        print(f"Loading from HuggingFace: {hf_dataset_path}")
        ds = load_dataset(hf_dataset_path)
        split_name = list(ds.keys())[0]
        data = list(ds[split_name])
    elif local_json_path:
        print(f"Loading from local file: {local_json_path}")
        with open(local_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError("Either hf_dataset_path or local_json_path must be provided")
    
    print(f"Total samples loaded: {len(data)}")
    
    # Split based on strategy
    if use_random_split:
        # Random split with specified validation_size
        train_data, test_data = train_test_split(
            data, 
            test_size=(1 - validation_size),
            random_state=20,
            shuffle=True
        )
        print(f"Validation samples: {len(train_data)} ({validation_size*100:.0f}% random split)")
        print(f"Test samples: {len(test_data)} ({(1-validation_size)*100:.0f}% random split)")
    else:
        # Sequential split: first 6 for validation, rest for test
        train_data = data[:6]
        test_data = data[6:]
        print(f"Validation samples: {len(train_data)} (first 6 questions)")
        print(f"Test samples: {len(test_data)} (remaining questions)")
    
    # Save validation set
    validation_path = os.path.join(output_dir, f"{dataset_name}_validate.jsonl")
    with open(validation_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved validation set to: {validation_path}")
    
    # Save test set
    test_path = os.path.join(output_dir, f"{dataset_name}_test.jsonl")
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved test set to: {test_path}")
    
    return validation_path, test_path


def main():
    """Main function to split all datasets"""
    print("\n" + "="*60)
    print("DATASET SPLITTING SCRIPT")
    print("Splitting datasets with random validation/test split (20%/80%)")
    print("="*60)
    
    # GAIA dataset (from local JSON file) - use random split
    # try:
    #     gaia_json = "data/datasets/GAIA.json"
    #     if os.path.exists(gaia_json):
    #         split_and_save_dataset(
    #             dataset_name="gaia",
    #             local_json_path=gaia_json,
    #             validation_size=0.2,
    #             use_random_split=True
    #         )
    #     else:
    #         print(f"\n⚠ WARNING: GAIA dataset not found at {gaia_json}")
    #         print("Please ensure GAIA.json exists in data/datasets/")
    # except Exception as e:
    #     print(f"\n✗ Error processing GAIA: {e}")
    
    # AIME24 dataset (from HuggingFace) - use random split
    try:
        split_and_save_dataset(
            dataset_name="aime24",
            hf_dataset_path="simplescaling/aime24_nofigures",
            validation_size=0.2,
            use_random_split=False
        )
    except Exception as e:
        print(f"\n✗ Error processing AIME24: {e}")
    
    # AIME25 dataset (from HuggingFace) - use random split
    # try:
    #     split_and_save_dataset(
    #         dataset_name="aime25",
    #         hf_dataset_path="simplescaling/aime25_nofigures",
    #         validation_size=0.2,
    #         use_random_split=True
    #     )
    # except Exception as e:
    #     print(f"\n✗ Error processing AIME25: {e}")
    
    print("\n" + "="*60)
    print("DATASET SPLITTING COMPLETED")
    print("="*60)
    print("\nGenerated files:")
    print("  - data/datasets/gaia_validate.jsonl")
    print("  - data/datasets/gaia_test.jsonl")
    print("  - data/datasets/aime24_validate.jsonl")
    print("  - data/datasets/aime24_test.jsonl")
    print("  - data/datasets/aime25_validate.jsonl")
    print("  - data/datasets/aime25_test.jsonl")
    print("\nYou can now use these files with the standard benchmark loading.")

if __name__ == "__main__":
    main()
