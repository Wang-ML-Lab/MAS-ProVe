"""
GAIA Benchmark Evaluation for LLM Debate System

IMPORTANT: Ensure the mas-process-eval judge server is running:
    cd /path/to/mas-process-eval
    python -m src.mas_proceval.servers.server_judge

The server must be running before evaluation starts.
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from datasets import load_dataset
from llm_debate_tool_call import DebateConfig, save_results
import cost_tracker
from cot import CoTConfig

from mas_debate import MASDebate
from mas_debate_iter import MASDebateIter
from mas_cot import MASCoT


class GAIAEvaluator:
    """Evaluator for the GAIA dataset using an LLM Debate System."""

    def __init__(self, subset: str = "test", output_dir: str = "results", process_eval: str = "agent"):
        """
        Initialize GAIA evaluator.

        Args:
            subset: 'test' or 'validation'
            output_dir: Base output directory for results
            process_eval: 'agent' or 'round' for process evaluation type
        """
        self.subset = subset
        self.dataset = None
        self.results_dir = os.path.join(output_dir, "gaia")
        os.makedirs(self.results_dir, exist_ok=True)
        self.process_eval = process_eval
    def load_dataset(self) -> bool:
        """Load the GAIA dataset from Hugging Face."""
        print(f"Loading GAIA {self.subset} dataset...")
        try:
            data_files={"test": "datasets/GAIA.json"}
            self.dataset = load_dataset("json", data_files=data_files, split=self.subset)
            print(f"Loaded {len(self.dataset)} examples")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def load_existing_results(self) -> Dict[str, Dict]:
        """Load existing results to skip already-processed task_ids."""
        result_file = f"gaia_{self.subset}_final.json"
        filepath = os.path.join(self.results_dir, result_file)
        
        if not os.path.exists(filepath):
            print("No existing results found.")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                existing = json.load(f)
            existing_map = {r["task_id"]: r for r in existing}
            print(f"Loaded {len(existing_map)} existing results from {result_file}")
            return existing_map
        except Exception as e:
            print(f"Failed to load results: {e}")
            return {}

    def format_question(self, example: Dict) -> str:
        """Format GAIA example into a question string."""
        return example.get("Question", "")
    
    def evaluate_answer(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Evaluate GAIA answer with flexible string matching."""
        import re
        
        # Extract <answer> tags if present
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            extracted_answer = response[start:end].strip()
        else:
            extracted_answer = response.strip()

        normalize = lambda x: str(x).lower().strip().replace(" ", "")
        norm_resp = normalize(extracted_answer)
        norm_expected = normalize(expected_answer)
        
        # Check if expected answer is numerical (after removing commas)
        norm_expected_no_comma = norm_expected.replace(",", "")
        is_numerical = re.match(r'^-?\d+\.?\d*$', norm_expected_no_comma) is not None
        
        if is_numerical:
            # For numerical answers, remove commas and do exact match
            norm_resp_no_comma = norm_resp.replace(",", "")
            flexible_match = norm_resp_no_comma == norm_expected_no_comma
        else:
            # For string answers, allow flexible matching
            exact_match = norm_resp == norm_expected
            
            # If response has parentheses, check parenthesized text
            paren_match = re.search(r'\(([^)]+)\)', extracted_answer)
            paren_content = normalize(paren_match.group(1)) if paren_match else ""
            
            # Check if expected answer appears as a substring
            substring_match = norm_expected in norm_resp or norm_expected in paren_content
            
            # Combined match logic for strings
            flexible_match = exact_match or substring_match

        return {
            "exact_match": flexible_match,
            "extracted_answer": extracted_answer,
            "expected_answer": expected_answer,
        }

    async def process_example_async(
        self, example: Dict, example_id: int, config, method: str = "debate"
    ) -> Dict:
        """Process a single GAIA example asynchronously."""
        task_id = example.get("task_id", f"gaia_{example_id}")
        print(f"Processing example {example_id + 1} ({example.get('Level', 'Unknown')}) with method={method}")
        
        if method == "cot":
            mas_cot = MASCoT(
                example, example_id, config, "gaia", "general",
                question_formatter=self.format_question,
                evaluate_fn=self.evaluate_answer
            )
            result = await mas_cot.run()
        elif self.process_eval == "round":
            mas_debate = MASDebateIter(
                example, example_id, config, "gaia", "general",
                question_formatter=self.format_question,
                evaluate_fn=self.evaluate_answer
            )
            result = await mas_debate.run()
        else:  # agent (default)
            mas_debate = MASDebate(
                example, example_id, config, "gaia", "general",
                question_formatter=self.format_question,
                evaluate_fn=self.evaluate_answer
            )
            result = await mas_debate.run()
        
        result["task_id"] = task_id
        result["level"] = example.get("Level")
        return result

    async def run_evaluation_async(
        self,
        config,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
        method: str = "debate"
    ) -> List[Dict]:
        """Run GAIA evaluation asynchronously."""
        if not self.load_dataset():
            return []

        existing_results = self.load_existing_results()

        if specific_ids:
            examples_with_ids = [(i, ex) for i, ex in enumerate(self.dataset) if i in specific_ids]
        else:
            examples_with_ids = [(i, ex) for i, ex in enumerate(list(self.dataset)[:max_examples])]

        # Filter out already-processed examples
        to_process = []
        for idx, ex in examples_with_ids:
            task_id = ex.get("task_id", f"gaia_{idx}")
            if task_id in existing_results:
                print(f"Skipping already processed task_id: {task_id}")
            else:
                to_process.append((idx, ex))

        if not to_process:
            print("All examples already processed. Returning existing results.")
            return list(existing_results.values())

        print(f"Running evaluation on {len(to_process)} new examples (skipped {len(examples_with_ids) - len(to_process)})...")

        tasks = [
            self.process_example_async(ex, idx, config, method)
            for idx, ex in to_process
        ]

        new_results = []
        batch_size = 10

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")

            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for j, res in enumerate(batch_results):
                if isinstance(res, Exception):
                    print(f"Error in example {i+j}: {res}")
                else:
                    new_results.append(res)
            
            # Save incrementally after each batch
            all_results = list(existing_results.values()) + new_results
            save_results(all_results, f"{self.results_dir}/gaia_{self.subset}_final.json")
            print(f"Saved {len(all_results)} results after batch {i//batch_size + 1}")

        # Final save and summary
        all_results = list(existing_results.values()) + new_results
        save_results(all_results, f"{self.results_dir}/gaia_{self.subset}_final.json")
        print(f"Processed {len(new_results)} new examples; total {len(all_results)}")
        self.print_summary(all_results)
        return all_results

    def run_evaluation(
        self,
        config,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
        method: str = "debate"
    ) -> List[Dict]:
        """Synchronous wrapper for async evaluation."""
        return asyncio.run(
            self.run_evaluation_async(config, max_examples, specific_ids, method)
        )

    def print_summary(self, results: List[Dict]):
        """Print evaluation summary."""
        if not results:
            print("No results to summarize.")
            return

        total_examples = len(results)
        
        # Handle both debate results (with agent_evaluations) and CoT results (with evaluation)
        if results[0].get("agent_evaluations"):
            # Debate format
            all_exact_matches = [res["evaluation"]["exact_match"] for r in results for res in r.get("agent_evaluations", [])]
        elif results[0].get("evaluation"):
            # CoT format (MASCoT)
            all_exact_matches = [r["evaluation"]["exact_match"] for r in results]
        else:
            # Fallback for unexpected format
            all_exact_matches = [False]
        
        total_cost = sum(r.get("cost", 0.0) for r in results)
        
        accuracy = sum(all_exact_matches) / len(all_exact_matches) if all_exact_matches else 0.0

        print("\n" + "=" * 50)
        print("GAIA EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Examples: {total_examples}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Total Cost: ${total_cost:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    config = DebateConfig(
        model="gpt-5-mini",
        num_agents=3,
        num_rounds=2,
    )

    evaluator = GAIAEvaluator(subset="test")
    evaluator.run_evaluation(config, max_examples=10) 