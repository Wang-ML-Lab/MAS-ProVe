#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 1/18/2026
# @Desc    : Analyze question-wise performance from CSV results

import pandas as pd
import argparse
import json
import sys
from pathlib import Path


def load_jsonl_questions(jsonl_paths, id_offset):
    """
    Load questions from JSONL file(s) and create a mapping.
    
    Args:
        jsonl_paths: List of paths to JSONL files
        id_offset: Value to subtract from ID to get question number (e.g., 59 for AIME24, -1 for AIME25)
        
    Returns:
        Dict mapping question text to (id, question_number)
    """
    questions = []
    
    for jsonl_path in jsonl_paths:
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        questions.append({
                            'id': data['id'],
                            'problem': data['problem'],
                            'answer': data.get('answer', '')
                        })
        except FileNotFoundError:
            print(f"Warning: JSONL file not found: {jsonl_path}")
        except Exception as e:
            print(f"Warning: Error reading {jsonl_path}: {e}")
    
    # Sort by ID to establish question number order
    questions.sort(key=lambda x: x['id'])
    
    # Create mapping: problem text -> (id, question_number)
    # question_num = id - id_offset
    problem_to_info = {}
    for q in questions:
        q_num = int(q['id']) - id_offset
        problem_to_info[q['problem']] = {
            'id': q['id'],
            'question_num': q_num,
            'answer': q['answer']
        }
    
    return problem_to_info, questions


def analyze_performance(csv_path: str, jsonl_paths: list, id_offset: int):
    """
    Analyze question-wise performance from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing results
        jsonl_paths: List of paths to JSONL files with question data
        id_offset: Value to subtract from ID to get question number
    """
    try:
        # Load question mapping from JSONL files
        problem_to_info, all_questions = load_jsonl_questions(jsonl_paths, id_offset)
        
        if not problem_to_info:
            print("Error: No questions loaded from JSONL files")
            return
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if 'question' not in df.columns or 'score' not in df.columns:
            print("Error: CSV must contain 'problem' and 'score' columns")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Match problems to get IDs and question numbers
        df['id'] = None
        df['question_num'] = None
        df['expected_answer'] = None
        
        for idx, row in df.iterrows():
            problem_text = row['question']
            if problem_text in problem_to_info:
                info = problem_to_info[problem_text]
                df.at[idx, 'id'] = info['id']
                df.at[idx, 'question_num'] = info['question_num']
                df.at[idx, 'expected_answer'] = info['answer']
        
        # Filter out unmatched problems
        unmatched = df[df['id'].isna()]
        if len(unmatched) > 0:
            print(f"Warning: {len(unmatched)} questions in CSV could not be matched to JSONL")
        
        df = df[df['id'].notna()].copy()
        
        if len(df) == 0:
            print("Error: No questions matched between CSV and JSONL files")
            return
        
        # Determine if answer is correct (score == 1.0 means correct)
        df['correct'] = df['score'] == 1.0
        
        # Create a set of question numbers that are in the CSV
        tested_questions = set(df['question_num'].astype(int))
        
        # Calculate statistics (only for tested questions)
        total_tested = len(df)
        correct_count = df['correct'].sum()
        accuracy = (correct_count / total_tested * 100) if total_tested > 0 else 0
        
        # Print results for all 30 questions
        min_q = min(problem_to_info[p]['question_num'] for p in problem_to_info)
        max_q = max(problem_to_info[p]['question_num'] for p in problem_to_info)
        
        for q_num in range(min_q, max_q + 1):
            if q_num in tested_questions:
                # Question was in CSV - show correct or incorrect
                row = df[df['question_num'] == q_num].iloc[0]
                status = "✓" if row['correct'] else "✗"
                print(f"Q{q_num}: {status}")
            else:
                # Question not in CSV - used for validation
                print(f"Q{q_num}: SKIP")
        
        # Print final accuracy (only for tested questions)
        print(f"\nAccuracy: {correct_count}/{total_tested} = {accuracy:.2f}%")
        
        # Return summary dictionary
        return {
            'total_tested': total_tested,
            'correct': int(correct_count),
            'accuracy': accuracy,
            'total_questions': len(all_questions)
        }
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error analyzing performance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Analyze question-wise performance from CSV results'
    )
    
    parser.add_argument('csv_file', type=str, help='Path to CSV file with results')
    parser.add_argument('--jsonl', '-j', nargs='+', required=True,
                        help='Path(s) to JSONL file(s) containing question data')
    parser.add_argument('--id-offset', type=int, default=59,
                        help='ID offset: question_num = id - offset (default: 59 for AIME24, use -1 for AIME25)')
    
    args = parser.parse_args()
    
    # Run analysis
    result = analyze_performance(args.csv_file, args.jsonl, args.id_offset)
    
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
