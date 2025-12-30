import random
import string
from collections import namedtuple
from typing import List

import re
import numpy as np
import pandas as pd
import datasets


def get_all_examples():
    import json
    import os
    
    # Get the directory where utils.py is located (adas/aime)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to 'adas', then into 'dataset'
    base_dir = os.path.join(os.path.dirname(current_dir), 'dataset')

    val_path = os.path.join(base_dir, 'gaia_validate.jsonl')
    test_path = os.path.join(base_dir, 'gaia_test.jsonl')
        
    # print(f"[DEBUG] Loading data from: {val_path}")

    examples = []
    # Load validation examples
    if os.path.exists(val_path):
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                # Each example should have 'Question' and 'answer' keys
                if 'Question' in ex and 'answer' in ex:
                    examples.append({'Question': ex['Question'], 'answer': ex['answer']})

    # Load test examples (if needed, e.g., for evaluation)
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                if 'Question' in ex and 'answer' in ex:
                    examples.append({'Question': ex['Question'], 'answer': ex['answer']})

    return examples

def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def score_gaia(target: any, prediction: any) -> bool:
    """
    Robust scoring for GAIA that handles:
    1. Numerical equivalence (35.00 == 35, 1,000 == 1000, $50 == 50)
    2. Soft string matching (Target "Paris" in Prediction "The answer is Paris")
    """
    # 1. Convert to strings strictly for processing
    tgt_str = str(target).strip()
    pred_str = str(prediction).strip()

    # --- HELPER: Try to parse a pure number ---
    def parse_number(s):
        # Remove currency symbols, percentage signs, and commas
        # Keep dots and negative signs. 
        # Note: This regex keeps digits, dots, and negative signs.
        clean = re.sub(r'[^\d.-]', '', s) 
        try:
            return float(clean)
        except ValueError:
            return None

    # 2. Numerical Comparison (Priority)
    tgt_num = parse_number(tgt_str)
    pred_num = parse_number(pred_str)

    # If BOTH can be parsed as numbers, compare their values
    if tgt_num is not None and pred_num is not None:
        # math.isclose handles floating point precision (35.0000001 vs 35)
        # rel_tol=1e-5 allows for tiny precision differences
        return math.isclose(tgt_num, pred_num, rel_tol=1e-5)

    # 3. String Comparison (Fallback)
    # Normalize: lowercase and normalize whitespace (turn tabs/newlines into single space)
    def normalize(s):
        return " ".join(s.lower().split())

    norm_tgt = normalize(tgt_str)
    norm_pred = normalize(pred_str)

    # A. Exact Match (Cleaned)
    if norm_tgt == norm_pred:
        return True

    # B. Soft Match (Target is a substring of Prediction)
    # This handles verbose agents: "The answer is Paris" (Target: "Paris")
    if norm_tgt in norm_pred:
        return True

    return False



def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"