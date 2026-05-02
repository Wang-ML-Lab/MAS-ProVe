#!/usr/bin/env python3
"""
Evaluate HumanEval completions for a subset of attempted task_ids.

This avoids the default assertion in human_eval's CLI that expects all
problems to be attempted.
"""

import argparse
import json
import os
import tempfile

from human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_file", help="Path to completions jsonl")
    parser.add_argument("--k", default="1,10,100", help="Comma-separated k list")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output", default=None, help="Optional output json path")
    args = parser.parse_args()

    if not os.path.exists(args.sample_file):
        raise FileNotFoundError(f"Sample file not found: {args.sample_file}")

    samples = list(stream_jsonl(args.sample_file))
    if len(samples) == 0:
        raise ValueError("No completions found in sample file.")

    attempted_task_ids = sorted({sample["task_id"] for sample in samples})

    all_problems = read_problems()
    subset_records = []
    for task_id in attempted_task_ids:
        if task_id not in all_problems:
            raise KeyError(f"Task id {task_id} not found in HumanEval problems.")
        record = {"task_id": task_id}
        record.update(all_problems[task_id])
        subset_records.append(record)

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
        problem_file = tf.name
    try:
        write_jsonl(problem_file, subset_records)
        results = evaluate_functional_correctness(
            sample_file=args.sample_file,
            k=k_values,
            n_workers=args.n_workers,
            timeout=args.timeout,
            problem_file=problem_file,
        )
    finally:
        if os.path.exists(problem_file):
            os.remove(problem_file)

    out_text = json.dumps(results, indent=2, sort_keys=True)
    print(out_text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")


if __name__ == "__main__":
    main()
