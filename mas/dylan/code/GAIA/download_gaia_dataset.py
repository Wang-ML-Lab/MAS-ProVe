"""
Split GAIA dataset into individual JSON files
Similar format to AIME: one file per question
"""
import json
import os

# Paths
INPUT_JSON = "/common/home/vv382/Datasets/GAIA.json"
OUTPUT_DIR = "/common/home/vv382/Datasets/GAIA"

def main():
    # Load the GAIA dataset
    print(f"Loading GAIA dataset from {INPUT_JSON}")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} problems")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Split into individual files (use index, not id field)
    for idx, item in enumerate(data):
        # Create simplified structure (keep only essential fields)
        problem_data = {
            'problem': item['Question'],
            'answer': item['answer'],
            'level': item['Level'],
            'task_id': item['task_id']
        }
        
        # Save to individual file using sequential numbering
        output_file = os.path.join(OUTPUT_DIR, f"{idx}.json")
        with open(output_file, 'w') as f:
            json.dump(problem_data, f, indent=2)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1} problems...")
    
    print(f"\n✓ Successfully split {len(data)} problems into {OUTPUT_DIR}/")
    print(f"  Files: 0.json to {len(data)-1}.json")

if __name__ == "__main__":
    main()
