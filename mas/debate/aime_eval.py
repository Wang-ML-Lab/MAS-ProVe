
"""
AIME Evaluation Module

IMPORTANT: Ensure the mas-process-eval judge server is running:
    cd /path/to/mas-process-eval
    python -m src.mas_proceval.servers.server_judge

The server must be running before evaluation starts.
"""

import json
import os
import asyncio
import re
import pandas as pd
from typing import List, Dict, Any
from datasets import load_dataset
from llm_debate_tool_call import DebateConfig, save_results
import cost_tracker

from mas_debate import MASDebate
from mas_debate_iter import MASDebateIter

class AIMEEvaluator:
    def __init__(self, benchmark: str = "aime24", output_dir: str = "results", process_eval: str = "agent"):
        """
        Initialize AIME evaluator.
        
        Args:
            benchmark: 'aime24' or 'aime25'
            output_dir: Base output directory for results
            process_eval: 'agent' or 'round' for process evaluation type
        """
        self.benchmark = benchmark
        self.dataset = None
        self.results_dir = os.path.join(output_dir, benchmark)
        os.makedirs(self.results_dir, exist_ok=True)
        self.existing_results = {}
        self.process_eval = process_eval

    def load_dataset(self):
        print(f"Loading {self.benchmark.upper()} dataset...")
        dataset_name = f"simplescaling/{self.benchmark}_nofigures"
        dataset = load_dataset(dataset_name)
        df = pd.DataFrame(dataset['train'])
        self.dataset = [row.to_dict() for _, row in df.iterrows()]
        print(f"Loaded {len(self.dataset)} questions from {self.benchmark.upper()} dataset")
        if self.dataset:
            print(f"Dataset fields: {list(self.dataset[0].keys())}")
        return True

    def load_existing_results(self):
        self.existing_results = {}
        result_file = f"{self.benchmark}_final.json"
        
        filepath = os.path.join(self.results_dir, result_file)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results = json.load(f)
            for result in results:
                problem_id = result.get("problem_id")
                if problem_id is not None:
                    self.existing_results[problem_id] = result
            print(f"Loaded {len(results)} existing results from {result_file}")
        print(f"Total existing results loaded: {len(self.existing_results)}")

    def get_missing_problem_ids(self, requested_ids=None):
        if requested_ids is None:
            requested_ids = list(range(len(self.dataset))) if self.dataset else []
        return [pid for pid in requested_ids if pid not in self.existing_results]

    def force_reprocess_problems(self, problem_ids: list):
        for problem_id in problem_ids:
            if problem_id in self.existing_results:
                del self.existing_results[problem_id]
                print(f"Removed problem {problem_id} from cache - will be reprocessed")

    def format_question(self, example: Dict) -> str:
        """Format AIME problem into a question string."""
        problem_text = example.get('problem', '')
        return f"Solve this AIME problem:\n\n{problem_text}\n\nAnswers are integers (0-999). Show your work."
    
    def extract_numerical_answer(self, text: str) -> int:
        """Extract numerical answer from AIME response."""
        if "<answer>" in text and "</answer>" in text:
            answer_section = text.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            answer_section = text
        numbers = re.findall(r'\b\d{1,3}\b', answer_section)
        if numbers:
            return int(numbers[-1])
        patterns = [r'(?:answer|result|solution) is (\d{1,3})',
                    r'(?:equals|=) (\d{1,3})',
                    r'therefore (\d{1,3})',
                    r'thus (\d{1,3})']
        for pattern in patterns:
            match = re.search(pattern, answer_section.lower())
            if match:
                return int(match.group(1))
        all_numbers = re.findall(r'\b\d{1,3}\b', text)
        return int(all_numbers[-1]) if all_numbers else -1

    def evaluate_answer(self, response: str, expected_answer) -> Dict[str, Any]:
        """Evaluate AIME answer."""
        extracted_answer = self.extract_numerical_answer(response)
        # Handle both string (aime24) and numerical (aime25) expected answers
        try:
            expected_answer = int(expected_answer)
        except (ValueError, TypeError):
            expected_answer = -1
        exact_match = extracted_answer == expected_answer
        valid_range = 0 <= extracted_answer <= 999 if extracted_answer >= 0 else False
        return {
            "score": 1.0 if exact_match else 0.0,
            "exact_match": exact_match,
            "extracted_answer": extracted_answer,
            "expected_answer": expected_answer,
            "valid_range": valid_range,
            "response_length": len(response)
        }

    
    async def process_problem_async(self, example: Dict, example_id: int, debate_config: DebateConfig) -> Dict:
        print(f"Processing problem {example_id + 1} with process_eval={self.process_eval}")
        
        if self.process_eval == "round":
            mas_debate = MASDebateIter(
                example, example_id, debate_config, self.benchmark, "math",
                question_formatter=self.format_question,
                evaluate_fn=self.evaluate_answer
            )
        else:  # agent (default)
            mas_debate = MASDebate(
                example, example_id, debate_config, self.benchmark, "math",
                question_formatter=self.format_question,
                evaluate_fn=self.evaluate_answer
            )
        
        return await mas_debate.run()

    async def run_evaluation_async(self, debate_config: DebateConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        if not self.load_dataset():
            return
        self.load_existing_results()
        if force_reprocess:
            self.force_reprocess_problems(force_reprocess)
        requested_ids = specific_ids if specific_ids else list(range(len(self.dataset)))
        if max_examples:
            requested_ids = requested_ids[:max_examples]
            print(f"Limiting to first {max_examples} problems (out of {len(self.dataset)} total)")
        missing_ids = self.get_missing_problem_ids(requested_ids)
        print(f"Need to process {len(missing_ids)} problems: {missing_ids}")
        if not missing_ids:
            print("All requested problems already have results.")
            return [self.existing_results[pid] for pid in requested_ids if pid in self.existing_results]
        examples_to_process = [(self.dataset[i], i) for i in missing_ids]
        results = []
        tasks = [self.process_problem_async(example, example_id, debate_config) for example, example_id in examples_to_process]
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(tasks) - 1) // batch_size + 1}")
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    # Print exception details instead of silently ignoring
                    print(f"ERROR: Task failed with exception: {type(result).__name__}: {result}")
                    import traceback
                    traceback.print_exception(type(result), result, result.__traceback__)
                else:
                    results.append(result)
            
            # Save incrementally after each batch
            all_results = list(self.existing_results.values()) + results
            all_results.sort(key=lambda x: x.get("problem_id", 0))
            save_results(all_results, f"{self.results_dir}/{self.benchmark}_final.json")
            print(f"Saved {len(all_results)} results after batch {i // batch_size + 1}")
        
        # Final save and summary
        all_results = list(self.existing_results.values()) + results
        all_results.sort(key=lambda x: x.get("problem_id", 0))
        save_results(all_results, f"{self.results_dir}/{self.benchmark}_final.json")
        print(f"Processed {len(results)} new problems; total {len(all_results)}")
        self.print_summary(all_results)
        return all_results

    def run_evaluation(self, debate_config: DebateConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        return asyncio.run(self.run_evaluation_async(debate_config, max_examples, specific_ids, force_reprocess))

    def print_summary(self, results: List[Dict]):
        if not results:
            print("No results to summarize")
            return
        total_problems = len(results)
        all_scores = [res["evaluation"]["score"] for r in results for res in r["agent_evaluations"]]
        exact_matches = [res["evaluation"]["exact_match"] for r in results for res in r["agent_evaluations"]]
        valid_answers = [res["evaluation"]["valid_range"] for r in results for res in r["agent_evaluations"]]
        
        # Accumulate total cost
        total_cost = sum(r.get("cost", 0.0) for r in results)
        
        print(f"\n{self.benchmark.upper()} SUMMARY: {total_problems} problems")
        print(f"Avg Score: {sum(all_scores)/len(all_scores):.3f}")
        print(f"Exact Match: {sum(exact_matches)/len(exact_matches):.3f}")
        print(f"Valid Answer: {sum(valid_answers)/len(valid_answers):.3f}")
        print(f"Total Cost: ${total_cost:.4f}")


# Backward compatibility aliases
AIME24Evaluator = AIMEEvaluator


if __name__ == "__main__":
    config = DebateConfig(
        model="gpt-5-mini",
        num_agents=3,
        num_rounds=2
    )
    evaluator = AIMEEvaluator(benchmark="aime24")
    results = evaluator.run_evaluation(config)