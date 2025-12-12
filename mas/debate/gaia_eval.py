"""
GAIA Benchmark Evaluation for LLM Debate System

IMPORTANT: If using judge_type="ranking" with use_judge_server=True (default),
ensure the mas-process-eval judge server is running:
    cd /path/to/mas-process-eval
    python -m src.servers.server_judge

The server must be running before evaluation starts.
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from datasets import load_dataset
# from llm_debate import DebateConfig, save_results, debate
from llm_debate_tool_call import DebateConfig, save_results, debate
from collections import Counter
import cost_tracker
from RoundWise.beam_search_prm import BeamSearchPRM, BeamSearchConfig
from RoundWise.greedy_search_prm import GreedySearchPRM, GreedySearchConfig
from AgentWise.greedy_search_prm import GreedySearchPRM as AgentWiseGreedySearchPRM, GreedySearchConfig as AgentWiseGreedySearchConfig
from AgentWise.beam_search_prm import BeamSearchPRM as AgentWiseBeamSearchPRM, BeamSearchConfig as AgentWiseBeamSearchConfig


class GAIAEvaluator:
    """Evaluator for the GAIA dataset using an LLM Debate System."""

    def __init__(self, subset: str = "test", output_dir: str = "results"):
        """
        Initialize GAIA evaluator.

        Args:
            subset: 'test' or 'validation'
            output_dir: Base output directory for results
        """
        self.subset = subset
        self.dataset = None
        self.results_dir = os.path.join(output_dir, "gaia")
        os.makedirs(self.results_dir, exist_ok=True)

    # ----------------------------
    # DATASET LOADING
    # ----------------------------
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
    # ----------------------------
    # EVALUATION
    # ----------------------------
    def evaluate_response(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Evaluate a model’s response against the expected answer."""
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
        
        import re
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
            
            # Check if expected answer appears as a substring (helps for 'Peng Li (Li Peng)')
            substring_match = norm_expected in norm_resp or norm_expected in paren_content
            
            # Combined match logic for strings
            flexible_match = exact_match or substring_match

        return {
            "exact_match": flexible_match,
            "extracted_answer": extracted_answer,
            "expected_answer": expected_answer,
        }

    # ----------------------------
    # SINGLE EXAMPLE PROCESSING
    # ----------------------------
    async def process_example_async(
        self, example: Dict, example_id: int, debate_config: DebateConfig
    ) -> Dict:
        """Process a single GAIA example asynchronously."""
        print(f"Processing example {example_id + 1} ({example.get('Level', 'Unknown')})")

        question = example.get("Question", "")

        debate_result, log = await debate(
            model=debate_config.model,
            question=question,
            num_agents=debate_config.num_agents,
            num_rounds=debate_config.num_rounds,
            dataset="gaia",
            use_tools=debate_config.use_tools,
            temperature=debate_config.temperature,
            max_tokens=debate_config.max_tokens,
        )

        all_rounds = log[0]["refine_results"]
        final_responses = all_rounds[-1]
        expected = example.get("answer", "")

        evaluations = []
        for j, resp in enumerate(final_responses):
            answer = (
                resp.split("<answer>")[-1].split("</answer>")[0].strip()
                if "<answer>" in resp
                else resp.strip().split("\n")[-1]
            )
            evaluations.append(
                {
                    "agent_id": j,
                    "answer": answer,
                    "evaluation": self.evaluate_response(answer, expected),
                }
            )

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
        
        # Extract web searches
        web_searches = debate_result.get("web_searches", [])
        
        return {
            "example_id": example_id,
            "question": question,
            "level": example.get("Level"),
            "task_id": example.get("task_id", f"gaia_{example_id}"),
            "debate_result": {
                "round_history": all_rounds,
                "final_responses": final_responses,
                "selected_response": debate_result["response"],
                "selected_answer": debate_result["answer"],
                "web_searches": web_searches,
            },
            "agent_evaluations": evaluations,
            "expected_answer": expected,
            "cost": total_cost,
        }

    # ----------------------------
    # ARCHIVE MECHANISM
    # ----------------------------
    def load_existing_results(self, beam_search=False, greedy_search=False, agentwise_greedy_search=False, agentwise_beam_search=False) -> Dict[str, Dict]:
        """Load existing results to skip already-processed task_ids."""
        if beam_search:
            archive_file = f"{self.results_dir}/gaia_{self.subset}_beam_search_final.json"
        elif greedy_search:
            archive_file = f"{self.results_dir}/gaia_{self.subset}_greedy_search_final.json"
        elif agentwise_greedy_search:
            archive_file = f"{self.results_dir}/gaia_{self.subset}_agentwise_greedy_search_final.json"
        elif agentwise_beam_search:
            archive_file = f"{self.results_dir}/gaia_{self.subset}_agentwise_beam_search_final.json"
        else:
            archive_file = f"{self.results_dir}/gaia_{self.subset}_final.json"
        
        if not os.path.exists(archive_file):
            print("No existing results found.")
            return {}
        
        try:
            with open(archive_file, 'r') as f:
                existing = json.load(f)
            existing_map = {r["task_id"]: r for r in existing}
            print(f"Loaded {len(existing_map)} existing results from archive.")
            return existing_map
        except Exception as e:
            print(f"Failed to load archive: {e}")
            return {}

    # ----------------------------
    # MAIN EVALUATION LOOP
    # ----------------------------
    async def run_evaluation_async(
        self,
        debate_config: DebateConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Run GAIA evaluation asynchronously with archive support."""
        if not self.load_dataset():
            return []

        # Load existing results
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
            self.process_example_async(ex, idx, debate_config)
            for idx, ex in to_process
        ]

        new_results = []
        batch_size = 50

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            print(
                f"Processing batch {i//batch_size + 1}/"
                f"{(len(tasks)-1)//batch_size + 1}"
            )

            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for j, res in enumerate(batch_results):
                if isinstance(res, Exception):
                    print(f"Error in example {i+j}: {res}")
                    raise res
                else:
                    new_results.append(res)
            
            # Save incrementally after each batch
            all_results = list(existing_results.values()) + new_results
            save_results(all_results, f"{self.results_dir}/gaia_{self.subset}_final.json")
            print(f"Saved {len(all_results)} results after batch {i//batch_size + 1}")

        # Final save and summary
        all_results = list(existing_results.values()) + new_results
        save_results(all_results, f"{self.results_dir}/gaia_{self.subset}_final.json")
        self.print_summary(all_results)
        return all_results

    def run_evaluation(
        self,
        debate_config: DebateConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Synchronous wrapper for async evaluation."""
        return asyncio.run(
            self.run_evaluation_async(debate_config, max_examples, specific_ids)
        )

    # ==================== Generic PRM Search Methods ====================
    
    async def _process_example_prm_async(
        self, example: Dict, example_id: int, prm_class, config, search_type: str
    ) -> Dict:
        """Generic method to process a single example using any PRM search."""
        print(f"Processing example {example_id + 1} with {search_type} ({example.get('Level', 'Unknown')})")

        question = example.get("Question", "")
        
        # Run PRM search
        prm = prm_class(config)
        result = await prm.search(question)
        
        expected = example.get("answer", "")
        
        # Build result dict based on search type
        result_key = search_type.lower().replace(" ", "_").replace("-", "_") + "_result"
        search_result = {
            "best_response": result.get('best_response'),
            "best_score": result.get('best_score'),
            "trajectory": result.get('best_trajectory'),
            "total_nodes_explored": result.get('total_nodes_explored'),
        }
        
        # Extract answer from final response
        final_response = result['best_response']
        answer = (
            final_response.split("<answer>")[-1].split("</answer>")[0].strip()
            if "<answer>" in final_response
            else final_response.strip().split("\n")[-1]
        )
        search_result["final_answer"] = answer
        
        # Add beam-search specific fields
        if "beam" in search_type.lower():
            final_candidate_evaluations = []
            
            # Check if this is AgentWise beam search (has all_final_responses)
            if "agentwise" in search_type.lower():
                # AgentWise beam search: evaluate all_final_responses (all agents × all beams)
                all_final_responses = result.get('all_final_responses', [])
                all_final_nodes = result.get('all_final_nodes', [])
                
                for i, response in enumerate(all_final_responses):
                    eval_result = self.evaluate_response(response, expected)
                    node = all_final_nodes[i] if i < len(all_final_nodes) else None
                    final_candidate_evaluations.append({
                        "response": response,
                        "score": node.score if node else 0.0,
                        "round": node.round if node else 0,
                        "evaluation": eval_result
                    })
            else:
                # RoundWise beam search: evaluate final_candidates
                for node in result.get('final_candidates', []):
                    eval_result = self.evaluate_response(node.response, expected)
                    final_candidate_evaluations.append({
                        "response": node.response,
                        "score": node.score,
                        "round": node.round,
                        "evaluation": eval_result
                    })
            
            search_result["final_candidates"] = final_candidate_evaluations
            search_result["num_correct_candidates"] = sum(1 for e in final_candidate_evaluations if e["evaluation"]["exact_match"])
            search_result["any_correct"] = search_result["num_correct_candidates"] > 0
        
        # Add agentwise-specific fields
        if "agentwise" in search_type.lower():
            search_result["best_agent_id"] = result.get('best_agent_id')
        
        # Evaluate the best response
        evaluation = self.evaluate_response(answer if "beam" not in search_type.lower() else result['best_response'], expected)
        
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
        
        task_id = example.get("task_id", f"gaia_{example_id}")
        
        return {
            "example_id": example_id,
            "task_id": task_id,
            "question": question,
            "level": example.get("Level", "Unknown"),
            result_key: search_result,
            "evaluation": evaluation,
            "expected_answer": expected,
            "cost": total_cost,
        }
    
    async def _run_prm_evaluation_async(
        self,
        prm_class,
        config,
        search_type: str,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Generic method to run evaluation using any PRM search."""
        if not self.load_dataset():
            return []

        # Determine which results to load based on search type
        load_kwargs = {}
        if "agentwise" in search_type.lower() and "beam" in search_type.lower():
            load_kwargs["agentwise_beam_search"] = True
            result_filename = f"gaia_{self.subset}_agentwise_beam_search_final.json"
            batch_size = 50
        elif "agentwise" in search_type.lower() and "greedy" in search_type.lower():
            load_kwargs["agentwise_greedy_search"] = True
            result_filename = f"gaia_{self.subset}_agentwise_greedy_search_final.json"
            batch_size = 50
        elif "beam" in search_type.lower():
            load_kwargs["beam_search"] = True
            result_filename = f"gaia_{self.subset}_beam_search_final.json"
            batch_size = 50
        elif "greedy" in search_type.lower():
            load_kwargs["greedy_search"] = True
            result_filename = f"gaia_{self.subset}_greedy_search_final.json"
            batch_size = 20
        else:
            result_filename = f"gaia_{self.subset}_final.json"
            batch_size = 50

        # Load existing results
        existing_results = self.load_existing_results(**load_kwargs)

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

        print(f"Running {search_type} evaluation on {len(to_process)} new examples...")

        tasks = [
            self._process_example_prm_async(ex, idx, prm_class, config, search_type)
            for idx, ex in to_process
        ]

        new_results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            print(
                f"Processing batch {i//batch_size + 1}/"
                f"{(len(tasks)-1)//batch_size + 1}"
            )

            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for j, res in enumerate(batch_results):
                if isinstance(res, Exception):
                    print(f"Error in example {i+j}: {res}")
                else:
                    new_results.append(res)
            
            # Save incrementally after each batch
            all_results = list(existing_results.values()) + new_results
            save_results(all_results, f"{self.results_dir}/{result_filename}")
            print(f"Saved {len(all_results)} results after batch {i//batch_size + 1}")

        # Merge with existing and final save
        all_results = list(existing_results.values()) + new_results
        save_results(all_results, f"{self.results_dir}/{result_filename}")
        self._print_prm_summary(all_results, search_type)
        return all_results
    
    def _print_prm_summary(self, results: List[Dict], search_type: str):
        """Generic method to print summary for any PRM search results."""
        if not results:
            print("No results to summarize.")
            return

        total_examples = len(results)
        
        # Get the result key dynamically
        result_key = search_type.lower().replace(" ", "_").replace("-", "_") + "_result"
        
        # Count correct examples
        correct_examples = sum(1 for res in results 
                              if res.get("evaluation", {}).get("exact_match", False))
        
        accuracy = correct_examples / total_examples if total_examples > 0 else 0.0
        total_cost = sum(res.get("cost", 0.0) for res in results)
        
        # Extract search-specific stats
        avg_final_judge_score = sum(res.get(result_key, {}).get("best_score", 0.0) 
                                    for res in results) / total_examples if total_examples > 0 else 0
        avg_nodes_explored = sum(res.get(result_key, {}).get("total_nodes_explored", 0) 
                                for res in results) / total_examples if total_examples > 0 else 0
        
        # Level breakdown
        level_counts = Counter(res.get("level", "Unknown") for res in results)
        level_correct = Counter()
        for res in results:
            if res.get("evaluation", {}).get("exact_match", False):
                level_correct[res.get("level", "Unknown")] += 1

        print("\n" + "=" * 60)
        print(f"GAIA {search_type.upper()} EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Examples: {total_examples}")
        
        # Beam search specific metrics
        if "beam" in search_type.lower():
            examples_with_any_correct = sum(1 for res in results 
                                           if res.get(result_key, {}).get("any_correct", False))
            total_correct_candidates = sum(res.get(result_key, {}).get("num_correct_candidates", 0) 
                                          for res in results)
            total_final_candidates = sum(len(res.get(result_key, {}).get("final_candidates", [])) 
                                        for res in results)
            
            print(f"Any Candidate Correct: {examples_with_any_correct}/{total_examples} ({examples_with_any_correct/total_examples:.3f})")
            print(f"Best Response Correct: {correct_examples}/{total_examples} ({accuracy:.3f})")
            print(f"Total Correct Candidates: {total_correct_candidates}/{total_final_candidates}")
            print(f"Avg Correct per Example: {total_correct_candidates/total_examples:.2f}")
        else:
            print(f"Correct: {correct_examples}/{total_examples}")
            print(f"Accuracy: {accuracy:.3f}")
        
        print(f"Avg Judge Score: {avg_final_judge_score:.2f}/5")
        print(f"Avg Nodes Explored: {avg_nodes_explored:.1f}")
        print(f"Total Cost: ${total_cost:.4f}")
        
        if level_counts:
            print()
            print("Level Breakdown:")
            for level in sorted(level_counts.keys()):
                level_total = level_counts[level]
                level_acc = level_correct[level] / level_total if level_total > 0 else 0.0
                print(f"  {level}: {level_correct[level]}/{level_total} ({level_acc:.3f})")
        
        print("=" * 60)
    
    # ==================== Specific PRM Method Wrappers ====================
    
    async def process_example_beam_search_async(
        self, example: Dict, example_id: int, beam_config: BeamSearchConfig
    ) -> Dict:
        """Process a single GAIA example using beam search PRM."""
        return await self._process_example_prm_async(example, example_id, BeamSearchPRM, beam_config, "Beam Search PRM")

    async def run_beam_search_evaluation_async(
        self,
        beam_config: BeamSearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Run GAIA evaluation asynchronously using beam search PRM."""
        return await self._run_prm_evaluation_async(BeamSearchPRM, beam_config, "Beam Search PRM", max_examples, specific_ids)

    def run_beam_search_evaluation(
        self,
        beam_config: BeamSearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Synchronous wrapper for beam search evaluation."""
        return asyncio.run(
            self.run_beam_search_evaluation_async(beam_config, max_examples, specific_ids)
        )

    async def process_example_greedy_search_async(
        self, example: Dict, example_id: int, greedy_config: GreedySearchConfig
    ) -> Dict:
        """Process a single GAIA example using greedy search PRM."""
        return await self._process_example_prm_async(example, example_id, GreedySearchPRM, greedy_config, "Greedy Search PRM")

    async def run_greedy_search_evaluation_async(
        self,
        greedy_config: GreedySearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Run GAIA evaluation using greedy search PRM."""
        return await self._run_prm_evaluation_async(GreedySearchPRM, greedy_config, "Greedy Search PRM", max_examples, specific_ids)

    def run_greedy_search_evaluation(
        self,
        greedy_config: GreedySearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Synchronous wrapper for greedy search evaluation."""
        return asyncio.run(
            self.run_greedy_search_evaluation_async(greedy_config, max_examples, specific_ids)
        )

    async def process_example_agentwise_greedy_search_async(
        self, example: Dict, example_id: int, greedy_config: AgentWiseGreedySearchConfig
    ) -> Dict:
        """Process a single GAIA example using agent-wise greedy search PRM."""
        return await self._process_example_prm_async(example, example_id, AgentWiseGreedySearchPRM, greedy_config, "AgentWise Greedy Search PRM")

    async def run_agentwise_greedy_search_evaluation_async(
        self,
        greedy_config: AgentWiseGreedySearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Run GAIA evaluation using agent-wise greedy search PRM."""
        return await self._run_prm_evaluation_async(AgentWiseGreedySearchPRM, greedy_config, "AgentWise Greedy Search PRM", max_examples, specific_ids)

    def run_agentwise_greedy_search_evaluation(
        self,
        greedy_config: AgentWiseGreedySearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Synchronous wrapper for agent-wise greedy search evaluation."""
        return asyncio.run(
            self.run_agentwise_greedy_search_evaluation_async(greedy_config, max_examples, specific_ids)
        )

    async def process_example_agentwise_beam_search_async(
        self, example: Dict, example_id: int, beam_config: AgentWiseBeamSearchConfig
    ) -> Dict:
        """Process a single GAIA example using agent-wise beam search PRM."""
        return await self._process_example_prm_async(example, example_id, AgentWiseBeamSearchPRM, beam_config, "AgentWise Beam Search PRM")

    async def run_agentwise_beam_search_evaluation_async(
        self,
        beam_config: AgentWiseBeamSearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Run GAIA evaluation using agent-wise beam search PRM."""
        return await self._run_prm_evaluation_async(AgentWiseBeamSearchPRM, beam_config, "AgentWise Beam Search PRM", max_examples, specific_ids)

    def run_agentwise_beam_search_evaluation(
        self,
        beam_config: AgentWiseBeamSearchConfig,
        max_examples: int = 20,
        specific_ids: list[int] | None = None,
    ) -> List[Dict]:
        """Synchronous wrapper for agent-wise beam search evaluation."""
        return asyncio.run(
            self.run_agentwise_beam_search_evaluation_async(beam_config, max_examples, specific_ids)
        )

    def print_agentwise_greedy_search_summary(self, results: List[Dict]):
        """Print evaluation summary for agent-wise greedy search results (backward compatibility wrapper)."""
        self._print_prm_summary(results, "AgentWise Greedy Search PRM")

    def print_agentwise_beam_search_summary(self, results: List[Dict]):
        """Print evaluation summary for agent-wise beam search results (backward compatibility wrapper)."""
        self._print_prm_summary(results, "AgentWise Beam Search PRM")

    def print_greedy_search_summary(self, results: List[Dict]):
        """Print evaluation summary for greedy search results (backward compatibility wrapper)."""
        self._print_prm_summary(results, "Greedy Search PRM")

    def print_beam_search_summary(self, results: List[Dict]):
        """Print evaluation summary for beam search results (backward compatibility wrapper)."""
        self._print_prm_summary(results, "Beam Search PRM")

    # ----------------------------
    # SUMMARY
    # ----------------------------
    def print_summary(self, results: List[Dict]):
        """Print evaluation summary."""
        if not results:
            print("No results to summarize.")
            return

        total_examples = len(results)
        correct_examples = 0
        total_cost = 0.0

        for res in results:
            # Check if any agent got it correct
            any_correct = any(ev["evaluation"]["exact_match"] for ev in res["agent_evaluations"])
            if any_correct:
                correct_examples += 1
            
            # Accumulate cost
            total_cost += res.get("cost", 0.0)

        accuracy = correct_examples / total_examples if total_examples > 0 else 0.0

        print("\n" + "=" * 50)
        print("GAIA EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Correct: {correct_examples}/{total_examples}")
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
