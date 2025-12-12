
import json
import os
import re
import asyncio
import pandas as pd
from typing import List, Dict, Any
from datasets import load_dataset
from llm_debate import DebateConfig, save_results, debate
import cost_tracker
from RoundWise.beam_search_prm import BeamSearchPRM, BeamSearchConfig
from RoundWise.greedy_search_prm import GreedySearchPRM, GreedySearchConfig
from AgentWise.greedy_search_prm import GreedySearchPRM as AgentWiseGreedySearchPRM, GreedySearchConfig as AgentWiseGreedySearchConfig
from AgentWise.beam_search_prm import BeamSearchPRM as AgentWiseBeamSearchPRM, BeamSearchConfig as AgentWiseBeamSearchConfig

class AIMEEvaluator:
    def __init__(self, benchmark: str = "aime24", output_dir: str = "results"):
        """
        Initialize AIME evaluator.
        
        Args:
            benchmark: 'aime24' or 'aime25'
            output_dir: Base output directory for results
        """
        self.benchmark = benchmark
        self.dataset = None
        self.results_dir = os.path.join(output_dir, benchmark)
        os.makedirs(self.results_dir, exist_ok=True)
        self.existing_results = {}

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

    def load_existing_results(self, beam_search=False, greedy_search=False, agentwise_greedy_search=False, agentwise_beam_search=False):
        self.existing_results = {}
        if beam_search:
            # Load beam search results only
            result_file = f"{self.benchmark}_beam_search_final.json"
        elif greedy_search:
            # Load greedy search results only
            result_file = f"{self.benchmark}_greedy_search_final.json"
        elif agentwise_greedy_search:
            # Load agentwise greedy search results only
            result_file = f"{self.benchmark}_agentwise_greedy_search_final.json"
        elif agentwise_beam_search:
            # Load agentwise beam search results only
            result_file = f"{self.benchmark}_agentwise_beam_search_final.json"
        else:
            # Load regular results only
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

    def extract_numerical_answer(self, text: str) -> int:
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

    def evaluate_response(self, response: str, expected_answer) -> Dict[str, Any]:
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
        print(f"Processing problem {example_id + 1}")
        problem_text = example.get('problem', '')
        question = f"Solve this AIME problem:\n\n{problem_text}\n\nAnswers are integers (0-999). Show your work."
        
        debate_result, log = await debate(
            model=debate_config.model,
            question=question,
            num_agents=debate_config.num_agents,
            num_rounds=debate_config.num_rounds,
            dataset=self.benchmark,
            temperature=debate_config.temperature,
            max_tokens=debate_config.max_tokens
        )
        all_round_responses = log[0]["refine_results"]
        final_responses = all_round_responses[-1]
        expected_answer = example.get('answer')
        evaluations = [{
            "agent_id": j,
            "answer": response,
            "evaluation": self.evaluate_response(response, expected_answer)
        } for j, response in enumerate(final_responses)]
        
        # Compute cost from usage data
        usage_list = debate_result.get("usage", [])
        total_cost = 0.0
        if usage_list:
            for usage in usage_list:
                if usage:
                    cost = cost_tracker.compute_cost(
                        debate_config.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    total_cost += cost
        
        return {
            "example_id": example_id,
            "problem_id": example_id,
            "problem": example['problem'],
            "debate_result": {
                "round_history": all_round_responses,
                "final_responses": final_responses,
                "selected_response": debate_result["response"],
                "selected_answer": debate_result["answer"],
            },
            "agent_evaluations": evaluations,
            "expected_answer": expected_answer,
            "cost": total_cost,
        }

    async def run_evaluation_async(self, debate_config: DebateConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        if not self.load_dataset():
            return
        self.load_existing_results()
        if force_reprocess:
            self.force_reprocess_problems(force_reprocess)
        requested_ids = specific_ids if specific_ids else list(range(len(self.dataset)))
        if max_examples:
            requested_ids = requested_ids[:max_examples]
        missing_ids = self.get_missing_problem_ids(requested_ids)
        print(f"Need to process {len(missing_ids)} problems: {missing_ids}")
        if not missing_ids:
            print("All requested problems already have results.")
            return [self.existing_results[pid] for pid in requested_ids if pid in self.existing_results]
        examples_to_process = [(self.dataset[i], i) for i in missing_ids]
        results = []
        tasks = [self.process_problem_async(example, example_id, debate_config) for example, example_id in examples_to_process]
        batch_size = 30
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(tasks) - 1) // batch_size + 1}")
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in batch_results:
                if not isinstance(result, Exception):
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

    # ==================== Generic PRM Search Methods ====================
    
    async def _process_problem_prm_async(self, example: Dict, example_id: int, prm_class, config, search_type: str) -> Dict:
        """Generic method to process a single problem using any PRM search"""
        print(f"Processing problem {example_id + 1} with {search_type}")
        problem_text = example.get('problem', '')
        question = f"Solve this AIME problem:\n\n{problem_text}\n\nAnswers are integers (0-999). Show your work."
        
        # Run PRM search
        prm = prm_class(config)
        result = await prm.search(question)
        
        expected_answer = example.get('answer')
        
        # Build result dict based on search type
        result_key = search_type.lower().replace(" ", "_").replace("-", "_") + "_result"
        search_result = {
            "best_response": result.get('best_response'),
            "best_score": result.get('best_score'),
            "trajectory": result.get('best_trajectory'),
            "total_nodes_explored": result.get('total_nodes_explored'),
        }
        
        # Add beam-search specific fields
        if "beam" in search_type.lower():
            # Evaluate ALL final responses (all agents from all beams)
            all_final_responses = result.get('all_final_responses', [])
            final_response_evaluations = []
            
            for response in all_final_responses:
                eval_result = self.evaluate_response(response, expected_answer)
                final_response_evaluations.append({
                    "response": response,
                    "evaluation": eval_result
                })
            
            search_result["all_final_response_evaluations"] = final_response_evaluations
            search_result["num_correct_responses"] = sum(1 for e in final_response_evaluations if e["evaluation"]["exact_match"])
            search_result["any_correct"] = search_result["num_correct_responses"] > 0
            search_result["total_final_responses"] = len(all_final_responses)
        
        # Add agentwise-specific fields
        if "agentwise" in search_type.lower():
            search_result["best_agent_id"] = result.get('best_agent_id')
        
        # Evaluate the best response
        best_evaluation = self.evaluate_response(result['best_response'], expected_answer)
        
        # Compute cost
        total_cost = 0.0
        for node in result.get('all_nodes', []):
            if node.usage:
                for usage in node.usage:
                    if usage:
                        cost = cost_tracker.compute_cost(
                            config.debate_config.model,
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0)
                        )
                        total_cost += cost
        
        # Add judge usage
        for usage in result.get('judge_usage', []):
            if usage:
                cost = cost_tracker.compute_cost(
                    config.judge_model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0)
                )
                total_cost += cost
        
        return {
            "example_id": example_id,
            "problem_id": example_id,
            "problem": example['problem'],
            result_key: search_result,
            "evaluation": best_evaluation,
            "expected_answer": expected_answer,
            "cost": total_cost,
        }
    
    async def _run_prm_evaluation_async(self, prm_class, config, search_type: str, max_examples=None, specific_ids=None, force_reprocess=None):
        """Generic method to run evaluation using any PRM search"""
        if not self.load_dataset():
            return
        
        # Determine which results to load based on search type
        load_kwargs = {}
        if "agentwise" in search_type.lower() and "beam" in search_type.lower():
            load_kwargs["agentwise_beam_search"] = True
            result_filename = f"{self.benchmark}_agentwise_beam_search_final.json"
        elif "agentwise" in search_type.lower() and "greedy" in search_type.lower():
            load_kwargs["agentwise_greedy_search"] = True
            result_filename = f"{self.benchmark}_agentwise_greedy_search_final.json"
        elif "beam" in search_type.lower():
            load_kwargs["beam_search"] = True
            result_filename = f"{self.benchmark}_beam_search_final.json"
        elif "greedy" in search_type.lower():
            load_kwargs["greedy_search"] = True
            result_filename = f"{self.benchmark}_greedy_search_final.json"
        else:
            result_filename = f"{self.benchmark}_final.json"
        
        self.load_existing_results(**load_kwargs)
        
        if force_reprocess:
            self.force_reprocess_problems(force_reprocess)
        
        requested_ids = specific_ids if specific_ids else list(range(len(self.dataset)))
        if max_examples:
            requested_ids = requested_ids[:max_examples]
        
        missing_ids = self.get_missing_problem_ids(requested_ids)
        print(f"Need to process {len(missing_ids)} problems with {search_type}: {missing_ids}")
        
        if not missing_ids:
            print("All requested problems already have results.")
            return [self.existing_results[pid] for pid in requested_ids if pid in self.existing_results]
        
        examples_to_process = [(self.dataset[i], i) for i in missing_ids]
        results = []
        
        tasks = [
            self._process_problem_prm_async(example, example_id, prm_class, config, search_type)
            for example, example_id in examples_to_process
        ]
        
        batch_size = 30
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(tasks) - 1) // batch_size + 1}")
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Error processing problem: {result}")
                else:
                    results.append(result)
            
            # Save incrementally after each batch
            all_results = list(self.existing_results.values()) + results
            all_results.sort(key=lambda x: x.get("problem_id", 0))
            save_results(all_results, f"{self.results_dir}/{result_filename}")
            print(f"Saved {len(all_results)} results after batch {i // batch_size + 1}")
        
        # Final save and summary
        all_results = list(self.existing_results.values()) + results
        all_results.sort(key=lambda x: x.get("problem_id", 0))
        
        save_results(all_results, f"{self.results_dir}/{result_filename}")
        print(f"Processed {len(results)} new problems with {search_type}; total {len(all_results)}")
        self._print_prm_summary(all_results, search_type)
        return all_results
    
    def _print_prm_summary(self, results: List[Dict], search_type: str):
        """Generic method to print summary for any PRM search results"""
        if not results:
            print("No results to summarize")
            return
        
        total_problems = len(results)
        correct_problems = sum(1 for r in results if r.get("evaluation", {}).get("exact_match", False))
        valid_answers = sum(1 for r in results if r.get("evaluation", {}).get("valid_range", False))
        avg_score = sum(r.get("evaluation", {}).get("score", 0.0) for r in results) / total_problems if total_problems > 0 else 0
        total_cost = sum(r.get("cost", 0.0) for r in results)
        
        # Get the result key dynamically
        result_key = search_type.lower().replace(" ", "_").replace("-", "_") + "_result"
        
        # Extract search-specific stats
        avg_final_judge_score = sum(r.get(result_key, {}).get("best_score", 0.0) for r in results) / total_problems if total_problems > 0 else 0
        avg_nodes_explored = sum(r.get(result_key, {}).get("total_nodes_explored", 0) for r in results) / total_problems if total_problems > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"{self.benchmark.upper()} {search_type.upper()} SUMMARY: {total_problems} problems")
        print(f"{'='*60}")
        
        # Beam search specific metrics
        if "beam" in search_type.lower():
            problems_with_any_correct = sum(1 for r in results if r.get(result_key, {}).get("any_correct", False))
            total_correct_responses = sum(r.get(result_key, {}).get("num_correct_responses", 0) for r in results)
            total_final_responses = sum(r.get(result_key, {}).get("total_final_responses", 0) for r in results)
            
            print(f"Any Response Correct: {problems_with_any_correct}/{total_problems} ({problems_with_any_correct/total_problems:.3f})")
            print(f"Best Response Correct: {correct_problems}/{total_problems} ({correct_problems/total_problems:.3f})")
            print(f"Total Correct Responses: {total_correct_responses}/{total_final_responses}")
            print(f"Avg Correct per Problem: {total_correct_responses/total_problems:.2f}")
        else:
            print(f"Correct: {correct_problems}/{total_problems} ({correct_problems/total_problems:.3f})")
        
        print(f"Valid Answer Rate: {valid_answers}/{total_problems} ({valid_answers/total_problems:.3f})")
        print(f"Avg Score: {avg_score:.3f}")
        print(f"Avg Judge Score: {avg_final_judge_score:.2f}/5")
        print(f"Avg Nodes Explored: {avg_nodes_explored:.1f}")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"{'='*60}")
    
    # ==================== Specific PRM Method Wrappers ====================
    
    async def process_problem_beam_search_async(self, example: Dict, example_id: int, beam_config: BeamSearchConfig) -> Dict:
        """Process a single problem using beam search PRM"""
        return await self._process_problem_prm_async(example, example_id, BeamSearchPRM, beam_config, "Beam Search PRM")

    async def run_beam_search_evaluation_async(self, beam_config: BeamSearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Run evaluation using beam search PRM"""
        return await self._run_prm_evaluation_async(BeamSearchPRM, beam_config, "Beam Search PRM", max_examples, specific_ids, force_reprocess)

    async def run_beam_search_evaluation(self, beam_config: BeamSearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Synchronous wrapper for beam search evaluation"""
        return asyncio.run(self.run_beam_search_evaluation_async(beam_config, max_examples, specific_ids, force_reprocess))

    async def process_problem_greedy_search_async(self, example: Dict, example_id: int, greedy_config: GreedySearchConfig) -> Dict:
        """Process a single problem using greedy search PRM"""
        return await self._process_problem_prm_async(example, example_id, GreedySearchPRM, greedy_config, "Greedy Search PRM")

    async def run_greedy_search_evaluation_async(self, greedy_config: GreedySearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Run evaluation using greedy search PRM"""
        return await self._run_prm_evaluation_async(GreedySearchPRM, greedy_config, "Greedy Search PRM", max_examples, specific_ids, force_reprocess)

    async def run_greedy_search_evaluation(self, greedy_config: GreedySearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Synchronous wrapper for greedy search evaluation"""
        return asyncio.run(self.run_greedy_search_evaluation_async(greedy_config, max_examples, specific_ids, force_reprocess))

    async def process_problem_agentwise_greedy_search_async(self, example: Dict, example_id: int, greedy_config: AgentWiseGreedySearchConfig) -> Dict:
        """Process a single problem using agent-wise greedy search PRM"""
        return await self._process_problem_prm_async(example, example_id, AgentWiseGreedySearchPRM, greedy_config, "AgentWise Greedy Search PRM")

    async def run_agentwise_greedy_search_evaluation_async(self, greedy_config: AgentWiseGreedySearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Run evaluation using agent-wise greedy search PRM"""
        return await self._run_prm_evaluation_async(AgentWiseGreedySearchPRM, greedy_config, "AgentWise Greedy Search PRM", max_examples, specific_ids, force_reprocess)

    async def run_agentwise_greedy_search_evaluation(self, greedy_config: AgentWiseGreedySearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Synchronous wrapper for agent-wise greedy search evaluation"""
        return asyncio.run(self.run_agentwise_greedy_search_evaluation_async(greedy_config, max_examples, specific_ids, force_reprocess))

    async def process_problem_agentwise_beam_search_async(self, example: Dict, example_id: int, beam_config: AgentWiseBeamSearchConfig) -> Dict:
        """Process a single problem using agent-wise beam search PRM"""
        return await self._process_problem_prm_async(example, example_id, AgentWiseBeamSearchPRM, beam_config, "AgentWise Beam Search PRM")

    async def run_agentwise_beam_search_evaluation_async(self, beam_config: AgentWiseBeamSearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Run evaluation using agent-wise beam search PRM"""
        return await self._run_prm_evaluation_async(AgentWiseBeamSearchPRM, beam_config, "AgentWise Beam Search PRM", max_examples, specific_ids, force_reprocess)

    def run_agentwise_beam_search_evaluation(self, beam_config: AgentWiseBeamSearchConfig, max_examples=None, specific_ids=None, force_reprocess=None):
        """Synchronous wrapper for agent-wise beam search evaluation"""
        return asyncio.run(self.run_agentwise_beam_search_evaluation_async(beam_config, max_examples, specific_ids, force_reprocess))

    def print_agentwise_greedy_search_summary(self, results: List[Dict]):
        """Print summary for agent-wise greedy search results (backward compatibility wrapper)"""
        self._print_prm_summary(results, "AgentWise Greedy Search PRM")

    def print_agentwise_beam_search_summary(self, results: List[Dict]):
        """Print summary for agent-wise beam search results (backward compatibility wrapper)"""
        self._print_prm_summary(results, "AgentWise Beam Search PRM")

    def print_greedy_search_summary(self, results: List[Dict]):
        """Print summary for greedy search results (backward compatibility wrapper)"""
        self._print_prm_summary(results, "Greedy Search PRM")

    def print_beam_search_summary(self, results: List[Dict]):
        """Print summary for beam search results (backward compatibility wrapper)"""
        self._print_prm_summary(results, "Beam Search PRM")

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
