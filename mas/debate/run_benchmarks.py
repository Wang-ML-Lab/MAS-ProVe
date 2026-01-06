"""
Unified Benchmark Runner for LLM Debate System
Supports AIME24/25 and GAIA evaluations with process evaluation
"""

import argparse
import os
from llm_debate_tool_call import DebateConfig
from cot import CoTConfig
from aime_eval import AIMEEvaluator
from gaia_eval import GAIAEvaluator


def run_aime24(config, max_examples: int = None, specific_ids: list = None, process_eval: str = "agent", method: str = "debate", output_dir: str = "results"):
    """Run AIME24 evaluation"""
    print("Starting AIME24 evaluation...")
    config.dataset = "aime24"
    if hasattr(config, 'use_tools'):
        config.use_tools = False  # AIME is pure math, no web search needed
    evaluator = AIMEEvaluator(benchmark="aime24", output_dir=output_dir, process_eval=process_eval)
    return evaluator.run_evaluation(config, max_examples, specific_ids, method=method)


def run_aime25(config, max_examples: int = None, specific_ids: list = None, process_eval: str = "agent", method: str = "debate", output_dir: str = "results"):
    """Run AIME25 evaluation"""
    print("Starting AIME25 evaluation...")
    config.dataset = "aime25"
    if hasattr(config, 'use_tools'):
        config.use_tools = False  # AIME is pure math, no web search needed
    evaluator = AIMEEvaluator(benchmark="aime25", output_dir=output_dir, process_eval=process_eval)
    return evaluator.run_evaluation(config, max_examples, specific_ids, method=method)


def run_gaia(config, max_examples: int = None, specific_ids: list = None, process_eval: str = "agent", method: str = "debate", output_dir: str = "results"):
    """Run GAIA evaluation"""
    print("Starting GAIA evaluation...")
    config.dataset = "gaia"
    if hasattr(config, 'use_tools'):
        config.use_tools = True  # GAIA often requires web search
    evaluator = GAIAEvaluator(subset="test", output_dir=output_dir, process_eval=process_eval)
    return evaluator.run_evaluation(config, max_examples, specific_ids, method=method)


def main():
    parser = argparse.ArgumentParser(description="Run LLM Debate benchmark evaluations")
    
    # Benchmark selection
    parser.add_argument("--benchmark", type=str, choices=["aime24", "aime25", "gaia", "all"], 
                       default="all", help="Which benchmark to run")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-5-mini", 
                       help="Model to use for debate")
    parser.add_argument("--num-agents", type=int, default=2, 
                       help="Number of agents in debate")
    parser.add_argument("--num-rounds", type=int, default=3, 
                       help="Number of debate rounds")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Temperature for model generation")
    
    # Process evaluation type
    parser.add_argument("--process-eval", type=str, choices=["agent", "round"], 
                       default="agent", help="Process evaluation type: 'agent' or 'round'")
    
    # Method selection
    parser.add_argument("--method", type=str, choices=["debate", "cot"], 
                       default="debate", help="Reasoning method: 'debate' or 'cot' (Chain of Thought)")
    parser.add_argument("--max-steps", type=int, default=4, 
                       help="Maximum steps for CoT (only used when --method cot)")
    
    # Evaluation limits
    parser.add_argument("--max-examples", type=int, default=None, 
                       help="Max examples to evaluate per benchmark")
    
    # Problem ID selection
    parser.add_argument("--problem-ids", type=int, nargs="+", 
                       help="Specific problem IDs to run (e.g., --problem-ids 1 3 5)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create configuration based on method
    if args.method == "cot":
        config = CoTConfig(
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            use_tools=False
        )
        print(f"Running Chain of Thought evaluation with configuration:")
        print(f"  Model: {config.model}")
        print(f"  Max Steps: {config.max_steps}")
        print(f"  Temperature: {config.temperature}")
    else:
        config = DebateConfig(
            model=args.model,
            num_agents=args.num_agents,
            num_rounds=args.num_rounds,
            temperature=args.temperature,
            use_tools=False
        )
        print(f"Running LLM Debate evaluation with configuration:")
        print(f"  Model: {config.model}")
        print(f"  Agents: {config.num_agents}")
        print(f"  Rounds: {config.num_rounds}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Process Eval: {args.process_eval}")
    
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Method: {args.method}")
    print()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    if args.benchmark in ["aime24", "all"]:
        print(f"\n{'='*60}")
        print("RUNNING AIME24 EVALUATION")
        if args.problem_ids:
            print(f"Specific Problem IDs: {args.problem_ids}")
        print(f"{'='*60}")
        
        results["aime24"] = run_aime24(
            config, 
            args.max_examples, 
            args.problem_ids,
            args.process_eval,
            args.method,
            args.output_dir
        )
    
    if args.benchmark in ["aime25", "all"]:
        print(f"\n{'='*60}")
        print("RUNNING AIME25 EVALUATION")
        if args.problem_ids:
            print(f"Specific Problem IDs: {args.problem_ids}")
        print(f"{'='*60}")
        
        results["aime25"] = run_aime25(
            config, 
            args.max_examples, 
            args.problem_ids,
            args.process_eval,
            args.method,
            args.output_dir
        )
    
    if args.benchmark in ["gaia", "all"]:
        print(f"\n{'='*60}")
        print("RUNNING GAIA EVALUATION")
        if args.problem_ids:
            print(f"Specific Problem IDs: {args.problem_ids}")
        print(f"{'='*60}")
        
        results["gaia"] = run_gaia(
            config, 
            args.max_examples, 
            args.problem_ids,
            args.process_eval,
            args.method,
            args.output_dir
        )
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved in: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main()