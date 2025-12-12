import json
import os
import asyncio
import re
from typing import List, Dict, Any
from datasets import load_dataset
from llm_debate import DebateConfig, debate, save_results
import cost_tracker


class HumanEvalEvaluator:
    """Evaluator for HumanEval dataset with pass@k metric"""

    def __init__(self):
        self.dataset = None
        self.results_dir = "results/humaneval"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_dataset(self):
        """Load HumanEval dataset"""
        print("Loading HumanEval dataset...")
        self.dataset = load_dataset("openai_humaneval", split="test")
        print(f"Loaded {len(self.dataset)} problems")
        return True

    def format_problem(self, example: Dict) -> tuple:
        """Format HumanEval problem as prompt"""
        prompt_text = example.get("prompt", "")
        task_id = example.get("task_id", "")
        
        question = f"""Complete the following Python function:

{prompt_text}

Provide only the complete function implementation. Your code should:
1. Include the function signature exactly as given
2. Implement the function body
3. Be syntactically correct Python code

Return your solution in a code block."""

        context = f"Task: {task_id}"
        return question, context

    def extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code in markdown code blocks
        code_block_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code block, try to extract lines that look like code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return response.strip()

    def evaluate_solution(self, task_id: str, completion: str, test: str, entry_point: str) -> Dict[str, Any]:
        """Execute test cases for a solution"""
        try:
            # Create test code
            test_code = completion + "\n" + test + "\n" + f"check({entry_point})"
            
            # Execute in isolated namespace
            namespace = {}
            exec(test_code, namespace)
            
            return {
                "passed": True,
                "error": None
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    async def process_example(self, example: Dict, example_id: int, config: DebateConfig) -> Dict:
        """Process a single HumanEval example"""
        task_id = example.get("task_id", f"task_{example_id}")
        print(f"Processing {task_id} ({example_id + 1})")

        question, context = self.format_problem(example)
        full_question = f"{question}\n\nContext: {context}"

        debate_result, log = await debate(
            model=config.model,
            question=full_question,
            num_agents=config.num_agents,
            num_rounds=config.num_rounds,
            dataset="humaneval",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        all_round_responses = log[0]["refine_results"]
        final_responses = all_round_responses[-1]

        # Extract code from each agent's response
        agent_solutions = []
        for j, response in enumerate(final_responses):
            code = self.extract_code(response)
            
            # Evaluate the solution
            evaluation = self.evaluate_solution(
                task_id=task_id,
                completion=code,
                test=example["test"],
                entry_point=example["entry_point"]
            )
            
            agent_solutions.append({
                "agent_id": j,
                "code": code,
                "passed": evaluation["passed"],
                "error": evaluation["error"]
            })

        # Compute cost
        usage_list = debate_result.get("usage", [])
        total_cost = 0.0
        if usage_list:
            for usage in usage_list:
                if usage:
                    cost = cost_tracker.compute_cost(
                        config.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    total_cost += cost

        return {
            "task_id": task_id,
            "example_id": example_id,
            "prompt": example["prompt"],
            "canonical_solution": example["canonical_solution"],
            "agent_solutions": agent_solutions,
            "debate_result": {
                "selected_response": debate_result["response"],
            },
            "cost": total_cost
        }

    def load_existing_results(self) -> Dict[str, Dict]:
        """Load existing results to skip already-processed tasks"""
        archive_file = os.path.join(self.results_dir, "humaneval_final.json")
        if not os.path.exists(archive_file):
            print("No existing results found.")
            return {}
        
        with open(archive_file, 'r') as f:
            existing = json.load(f)
        
        existing_map = {r["task_id"]: r for r in existing}
        print(f"Loaded {len(existing_map)} existing results")
        return existing_map

    async def run_evaluation_async(self, config: DebateConfig, max_examples: int = None):
        """Run evaluation with pass@k metric"""
        if not self.load_dataset():
            return

        # Load existing results
        existing_results = self.load_existing_results()

        # Determine examples to process
        all_examples = list(self.dataset)
        if max_examples:
            all_examples = all_examples[:max_examples]

        to_process = []
        for idx, ex in enumerate(all_examples):
            task_id = ex.get("task_id", f"task_{idx}")
            if task_id in existing_results:
                print(f"Skipping {task_id}")
            else:
                to_process.append((idx, ex))

        if not to_process:
            print("All examples already processed")
            all_results = list(existing_results.values())
            self.print_summary(all_results, config.num_agents)
            return all_results

        print(f"\nProcessing {len(to_process)} examples (skipped {len(all_examples) - len(to_process)})\n")

        # Process examples
        tasks = [self.process_example(ex, idx, config) for idx, ex in to_process]
        new_results = []
        
        batch_size = 50
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            print(f"Batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for res in batch_results:
                if isinstance(res, Exception):
                    print(f"Error: {res}")
                else:
                    new_results.append(res)

        # Merge and save
        all_results = list(existing_results.values()) + new_results
        all_results.sort(key=lambda x: x.get("example_id", 0))
        
        save_results(all_results, os.path.join(self.results_dir, "humaneval_final.json"))
        
        print(f"\nProcessed {len(new_results)} new examples")
        self.print_summary(all_results, config.num_agents)
        
        return all_results

    def run_evaluation(self, config: DebateConfig, max_examples: int = None):
        """Synchronous wrapper"""
        return asyncio.run(self.run_evaluation_async(config, max_examples))

    def compute_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Compute pass@k metric.
        n: total number of samples
        c: number of correct samples
        k: k in pass@k
        """
        if n - c < k:
            return 1.0
        return 1.0 - (self._comb(n - c, k) / self._comb(n, k))

    def _comb(self, n: int, k: int) -> float:
        """Compute binomial coefficient"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def print_summary(self, results: List[Dict], num_agents: int):
        """Print evaluation summary with pass@k metrics"""
        if not results:
            print("No results to summarize")
            return

        total_tasks = len(results)
        total_passed = 0
        total_samples = 0
        total_cost = 0.0

        # Count passes per task
        task_passes = []
        for result in results:
            passed_agents = sum(1 for sol in result["agent_solutions"] if sol["passed"])
            task_passes.append(passed_agents)
            total_passed += passed_agents
            total_samples += len(result["agent_solutions"])
            total_cost += result.get("cost", 0.0)

        # Compute pass@k for k=1,2,3,...,num_agents
        print(f"\nHUMANEVAL SUMMARY ({total_tasks} tasks)")
        print(f"Total samples: {total_samples}")
        print(f"Passed samples: {total_passed}")
        
        for k in range(1, min(num_agents + 1, 6)):
            pass_at_k_scores = []
            for c in task_passes:
                score = self.compute_pass_at_k(num_agents, c, k)
                pass_at_k_scores.append(score)
            
            avg_pass_at_k = sum(pass_at_k_scores) / len(pass_at_k_scores)
            print(f"pass@{k}: {avg_pass_at_k:.3f}")
        
        print(f"Total Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    config = DebateConfig(
        model="gpt-5-mini",
        num_agents=3,
        num_rounds=2,
    )
    
    evaluator = HumanEvalEvaluator()
    results = evaluator.run_evaluation(config, max_examples=10)
