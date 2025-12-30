import random
import string
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
import datasets


def get_all_examples(dataset_name='aime24'):
    import json
    import os
    
    # Get the directory where utils.py is located (adas/aime)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to 'adas', then into 'dataset'
    base_dir = os.path.join(os.path.dirname(current_dir), 'dataset')
    
    # Select filename based on argument
    if dataset_name == 'aime25':
        val_path = os.path.join(base_dir, 'aime25_validate.jsonl')
        test_path = os.path.join(base_dir, 'aime25_test.jsonl')
    else:
        val_path = os.path.join(base_dir, 'aime24_validate.jsonl')
        test_path = os.path.join(base_dir, 'aime24_test.jsonl')
    
    print(f"[DEBUG] Loading data from: {val_path}")

    examples = []
    # Load validation examples
    if os.path.exists(val_path):
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                # Each example should have 'problem' and 'answer' keys
                if 'problem' in ex and 'answer' in ex:
                    examples.append({'problem': ex['problem'], 'answer': ex['answer']})

    # Load test examples (if needed, e.g., for evaluation)
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                if 'problem' in ex and 'answer' in ex:
                    examples.append({'problem': ex['problem'], 'answer': ex['answer']})

    return examples

def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id

import re

def score_aime(target: any, prediction: any) -> bool:
    """
    Scores whether the prediction matches the target exactly (numeric comparison).
    """
    try:
        # 1. Clean up target
        target_int = int(float(str(target).strip()))
        
        # 2. Clean up prediction (remove non-digits, handle 'Answer: 120' formats)
        pred_str = str(prediction)
        
        # Extract the last number found in the string
        # This handles cases like "The answer is 120."
        numbers = re.findall(r'-?\d+', pred_str)
        if not numbers:
            return False
            
        prediction_int = int(numbers[-1])
        # print(f"Scoring target: {target_int} with prediction: {prediction_int}")
        return target_int == prediction_int
    except Exception:
        print(f"Error scoring target: {target} with prediction: {prediction}")
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