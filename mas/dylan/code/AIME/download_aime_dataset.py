#!/usr/bin/env python3
"""
Download and prepare AIME 2024 dataset for DyLAN experiments
"""
import os
import json
from datasets import load_dataset


def download_aime_dataset():
    """
    Download AIME 2024 problems from Hugging Face
    Dataset: simplescaling/aime24_nofigures
    """
    
    base_dir = "/common/home/vv382/Datasets/AIME"
    os.makedirs(base_dir, exist_ok=True)
    
    print("="*60)
    print("AIME 2024 Dataset Download")
    print("="*60)
    print()
    
    print("Downloading AIME 2025 from Hugging Face...")
    print("Dataset: simplescaling/aime25_nofigures")
    
    # Load the dataset
    ds = load_dataset("simplescaling/aime25_nofigures")
    
    print(f"Available splits: {list(ds.keys())}")
    
    # The dataset only has 'train' split with all 30 questions
    train_data = ds['train']
    total_problems = len(train_data)
    
    print(f"Total problems: {total_problems}")
    print()
    
    # Create 2024 directory
    year_dir = os.path.join(base_dir, "2025")
    os.makedirs(year_dir, exist_ok=True)
    
    # Process all problems
    for idx, item in enumerate(train_data):
        problem_text = item['problem']
        # solution_text = item['solution']
        answer = str(item['answer'])
        
        # Create simple JSON with only required fields
        problem_data = {
            "problem": problem_text,
            # "solution": solution_text,
            "answer": answer
        }
        
        # Save with idx as filename (0-indexed for 0-29)
        filepath = os.path.join(year_dir, f"{idx}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(problem_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved Problem {idx+1}/30 → {filepath}")
    
    print()
    print(f"✓ AIME 2024 dataset downloaded to {year_dir}")
    print(f"  Total: {total_problems} problems")
    print()
    return year_dir

if __name__ == "__main__":
    year_dir = download_aime_dataset()
    
    print()
    print("="*60)
    print("Setup Complete!")
    print("="*60)
    print()
    print("Dataset structure:")
    print(f"  {year_dir}/")
    print(f"    0.json  → Problem 1")
    print(f"    1.json  → Problem 2")
    print(f"    ...")
    print(f"    29.json → Problem 30")
    print()
    print("To test the framework:")
    print("  cd /common/home/vv382/DyLAN-PRM/code/AIME")
    print("  export OPENAI_API_KEY='your-key'")
    print()
    print("Test single problem:")
    print(f"  python llmlp_gen_aime.py {year_dir} 0 0 gpt-4 gpt-4")
    print()
    print("Run all AIME problems (0-29):")
    print(f"  python llmlp_gen_aime.py {year_dir} 0 29 gpt-4 gpt-4")
    print()
    print("Run all 30 problems:")
    print("  bash exp_aime.sh")
    print()

