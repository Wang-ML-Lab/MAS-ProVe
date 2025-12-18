"""
Unified Benchmark Runner for LLM Debate System
Supports GAIA and AIME24/25 evaluations
- Decorator-based greedy search (main feature - uses client/server judge)
- Beam search (for experimentation)
"""

import argparse
import os
import asyncio
from llm_debate_tool_call import DebateConfig
from gaia_eval import GAIAEvaluator
from aime_eval import AIMEEvaluator
from RoundWise.beam_search_prm import BeamSearchConfig
from RoundWise.greedy_search_prm.greedy_search_decorator import GreedySearchConfig
from AgentWise.greedy_search_prm.greedy_search_decorator import GreedySearchConfig as AgentWiseGreedySearchConfig
from AgentWise.beam_search_prm import BeamSearchConfig as AgentWiseBeamSearchConfig


def run_gaia(config: DebateConfig, max_examples: int = 20):
    """Run GAIA evaluation"""
    print("Starting GAIA evaluation...")
    config.dataset = "gaia"
    evaluator = GAIAEvaluator(subset="test")
    return evaluator.run_evaluation(config, max_examples)


def run_aime24(config: DebateConfig, max_examples: int = None, specific_ids: list = None):
    """Run AIME24 evaluation"""
    print("Starting AIME24 evaluation...")
    config.dataset = "aime24"
    evaluator = AIMEEvaluator(benchmark="aime24")
    return evaluator.run_evaluation(config, max_examples, specific_ids)


def run_aime25(config: DebateConfig, max_examples: int = None, specific_ids: list = None):
    """Run AIME25 evaluation"""
    print("Starting AIME25 evaluation...")
    config.dataset = "aime25"
    evaluator = AIMEEvaluator(benchmark="aime25")
    return evaluator.run_evaluation(config, max_examples, specific_ids)


# ==================== Generic Helper Functions ====================

def _create_search_config(config_class, config: DebateConfig, judge_type: str,
                         prm_model_path: str = None, prm_api_url: str = None,
                         enable_summarization: bool = False, **search_params):
    """Generic helper to create any search config with PRM support"""
    search_config = config_class(
        debate_config=config,
        judge_model=config.judge_model if hasattr(config, 'judge_model') and config.judge_model else config.model,
        judge_type=judge_type,
        **search_params
    )
    
    # Add PRM-specific config if using PRM
    if judge_type == "prm":
        if prm_model_path:
            search_config.prm_model_path = prm_model_path
        if prm_api_url:
            search_config.prm_api_url = prm_api_url
        search_config.enable_summarization = enable_summarization
    
    return search_config


def _create_decorator_search_config(config_class, config: DebateConfig, task_type: str = "math",
                                   judge_server_host: str = "127.0.0.1",
                                   judge_server_port: int = 5555, **search_params):
    """Helper to create decorator-based search configs (no judge_model/judge_type needed)"""
    search_config = config_class(
        debate_config=config,
        task_type=task_type,
        judge_server_host=judge_server_host,
        judge_server_port=judge_server_port,
        **search_params
    )
    return search_config


def _run_search_evaluation(benchmark: str, search_type: str, config_class, 
                          eval_method_name: str, config: DebateConfig,
                          max_examples: int = None, specific_ids: list = None,
                          judge_type: str = "scoring", prm_model_path: str = None,
                          prm_api_url: str = None, enable_summarization: bool = False,
                          output_dir: str = "results",
                          **search_params):
    """Generic helper to run any search evaluation on any benchmark"""
    print(f"Starting {benchmark.upper()} with {search_type}...")
    config.dataset = benchmark
    
    # Create search config
    search_config = _create_search_config(
        config_class, config, judge_type, prm_model_path,
        prm_api_url, enable_summarization, **search_params
    )
    
    # Create evaluator and run
    if benchmark == "gaia":
        evaluator = GAIAEvaluator(subset="test", output_dir=output_dir)
        return getattr(evaluator, eval_method_name)(search_config, max_examples)
    elif benchmark in ["aime24", "aime25"]:
        evaluator = AIMEEvaluator(benchmark=benchmark, output_dir=output_dir)
        return getattr(evaluator, eval_method_name)(search_config, max_examples, specific_ids)


# ==================== Beam Search Functions ====================

def run_aime24_beam_search(config: DebateConfig, max_examples: int = None, 
                           specific_ids: list = None, beam_width: int = 3, 
                           beam_rounds: int = 2, judge_type: str = "scoring",
                           prm_model_path: str = None, prm_api_url: str = None,
                           enable_summarization: bool = False, output_dir: str = "results"):
    """Run AIME24 with beam search process evaluation"""
    return _run_search_evaluation(
        "aime24", "Beam Search PRM", BeamSearchConfig, 
        "run_beam_search_evaluation", config, max_examples, specific_ids,
        judge_type, prm_model_path, prm_api_url, enable_summarization,
        output_dir,
        beam_width=beam_width, num_rounds=beam_rounds
    )


def run_gaia_beam_search(config: DebateConfig, max_examples: int = 20, 
                         beam_width: int = 3, beam_rounds: int = 2, judge_type: str = "scoring",
                         prm_model_path: str = None, prm_api_url: str = None,
                         enable_summarization: bool = False, output_dir: str = "results"):
    """Run GAIA with beam search process evaluation"""
    return _run_search_evaluation(
        "gaia", "Beam Search PRM", BeamSearchConfig,
        "run_beam_search_evaluation", config, max_examples, None,
        judge_type, prm_model_path, prm_api_url, enable_summarization,
        output_dir,
        beam_width=beam_width, num_rounds=beam_rounds
    )


# ==================== Greedy Search Functions ====================

def run_aime24_greedy_search(config: DebateConfig, max_examples: int = None, 
                             specific_ids: list = None, num_candidates: int = 3, 
                             greedy_rounds: int = 2, judge_type: str = "scoring",
                             prm_model_path: str = None, prm_api_url: str = None,
                             enable_summarization: bool = False, output_dir: str = "results"):
    """Run AIME24 with greedy search process evaluation (decorator version)"""
    print(f"Starting AIME24 with Greedy Search PRM (Decorator)...")
    config.dataset = "aime24"
    
    # Create decorator-based config (no judge_model/judge_type/num_candidates needed)
    search_config = _create_decorator_search_config(
        GreedySearchConfig, config, task_type="math",
        num_rounds=greedy_rounds
    )
    
    # Create evaluator and run
    from aime_eval import AIMEEvaluator
    evaluator = AIMEEvaluator(benchmark="aime24", output_dir=output_dir)
    return evaluator.run_greedy_search_evaluation(search_config, max_examples, specific_ids)


def run_gaia_greedy_search(config: DebateConfig, max_examples: int = 20, 
                           num_candidates: int = 3, greedy_rounds: int = 2, judge_type: str = "scoring",
                           prm_model_path: str = None, prm_api_url: str = None,
                           enable_summarization: bool = False, output_dir: str = "results"):
    """Run GAIA with greedy search process evaluation (decorator version)"""
    print(f"Starting GAIA with Greedy Search PRM (Decorator)...")
    config.dataset = "gaia"
    
    # Create decorator-based config (no judge_model/judge_type/num_candidates needed)
    search_config = _create_decorator_search_config(
        GreedySearchConfig, config, task_type="general",
        num_rounds=greedy_rounds
    )
    
    # Create evaluator and run
    from gaia_eval import GAIAEvaluator
    evaluator = GAIAEvaluator(subset="test", output_dir=output_dir)
    return evaluator.run_greedy_search_evaluation(search_config, max_examples)


# ==================== AgentWise Greedy Search Functions ====================

def run_aime24_agentwise_greedy_search(config: DebateConfig, max_examples: int = None, 
                                       specific_ids: list = None, num_candidates: int = 3, 
                                       greedy_rounds: int = 2, judge_type: str = "scoring",
                                       prm_model_path: str = None, prm_api_url: str = None,
                                       enable_summarization: bool = False, output_dir: str = "results"):
    """Run AIME24 with agent-wise greedy search process evaluation (decorator version)"""
    print(f"Starting AIME24 with AgentWise Greedy Search PRM (Decorator)...")
    config.dataset = "aime24"
    
    # Create decorator-based config (no judge_model/judge_type/num_candidates needed)
    search_config = _create_decorator_search_config(
        AgentWiseGreedySearchConfig, config, task_type="math",
        num_rounds=greedy_rounds
    )
    
    # Create evaluator and run
    from aime_eval import AIMEEvaluator
    evaluator = AIMEEvaluator(benchmark="aime24", output_dir=output_dir)
    return evaluator.run_agentwise_greedy_search_evaluation(search_config, max_examples, specific_ids)


def run_gaia_agentwise_greedy_search(config: DebateConfig, max_examples: int = 20, 
                                     num_candidates: int = 3, greedy_rounds: int = 2, 
                                     judge_type: str = "scoring",
                                     prm_model_path: str = None, prm_api_url: str = None,
                                     enable_summarization: bool = False, output_dir: str = "results"):
    """Run GAIA with agent-wise greedy search process evaluation (decorator version)"""
    print(f"Starting GAIA with AgentWise Greedy Search PRM (Decorator)...")
    config.dataset = "gaia"
    
    # Create decorator-based config (no judge_model/judge_type/num_candidates needed)
    search_config = _create_decorator_search_config(
        AgentWiseGreedySearchConfig, config, task_type="general",
        num_rounds=greedy_rounds
    )
    
    # Create evaluator and run
    from gaia_eval import GAIAEvaluator
    evaluator = GAIAEvaluator(subset="test", output_dir=output_dir)
    return evaluator.run_agentwise_greedy_search_evaluation(search_config, max_examples)


# ==================== AgentWise Beam Search Functions ====================

def run_aime24_agentwise_beam_search(config: DebateConfig, max_examples: int = None, 
                                     specific_ids: list = None, beam_width: int = 3, 
                                     beam_rounds: int = 2, judge_type: str = "scoring",
                                     prm_model_path: str = None, prm_api_url: str = None,
                                     enable_summarization: bool = False, output_dir: str = "results"):
    """Run AIME24 with agent-wise beam search process evaluation"""
    return _run_search_evaluation(
        "aime24", "AgentWise Beam Search PRM", AgentWiseBeamSearchConfig,
        "run_agentwise_beam_search_evaluation", config, max_examples, specific_ids,
        judge_type, prm_model_path, prm_api_url, enable_summarization,
        output_dir,
        beam_width=beam_width, num_rounds=beam_rounds
    )


def run_gaia_agentwise_beam_search(config: DebateConfig, max_examples: int = 20, 
                                   beam_width: int = 3, beam_rounds: int = 2, 
                                   judge_type: str = "scoring",
                                   prm_model_path: str = None, prm_api_url: str = None,
                                   enable_summarization: bool = False, output_dir: str = "results"):
    """Run GAIA with agent-wise beam search process evaluation"""
    return _run_search_evaluation(
        "gaia", "AgentWise Beam Search PRM", AgentWiseBeamSearchConfig,
        "run_agentwise_beam_search_evaluation", config, max_examples, None,
        judge_type, prm_model_path, prm_api_url, enable_summarization,
        output_dir,
        beam_width=beam_width, num_rounds=beam_rounds
    )


def main():
    parser = argparse.ArgumentParser(description="Run LLM Debate benchmark evaluations")
    
    # Benchmark selection
    parser.add_argument("--benchmark", type=str, choices=["gaia", "aime24", "aime25", "all"], 
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
    
    # Evaluation limits
    parser.add_argument("--max-swe-examples", type=int, default=5, 
                       help="Max examples for SWE-bench")
    parser.add_argument("--max-gaia-examples", type=int, default=10, 
                       help="Max examples for GAIA")
    parser.add_argument("--max-aime-examples", type=int, default=None, 
                       help="Max examples for AIME24")
    parser.add_argument("--max-aime25-examples", type=int, default=None, 
                       help="Max examples for AIME25")
    
    # AIME24 specific selection
    parser.add_argument("--aime-problem-ids", type=int, nargs="+", 
                       help="Specific AIME problem IDs to run (e.g., --aime-problem-ids 1 3 5)")
    parser.add_argument("--aime25-problem-ids", type=int, nargs="+", 
                       help="Specific AIME25 problem IDs to run (e.g., --aime25-problem-ids 1 3 5)")
    
    # Web search / tools
    parser.add_argument("--gaia-use-tools", action="store_true", default=True,
                       help="Enable web search for GAIA (default: True)")
    parser.add_argument("--gaia-no-tools", dest="gaia_use_tools", action="store_false",
                       help="Disable web search for GAIA")
    
    # Beam search configuration
    parser.add_argument("--beam-search", action="store_true", default=False,
                       help="Use beam search process evaluation (AIME24,25 and GAIA only)")
    parser.add_argument("--beam-width", type=int, default=3,
                       help="Beam width for beam search (default: 3)")
    parser.add_argument("--beam-rounds", type=int, default=3,
                       help="Number of rounds for beam search (default: 3)")
    
    # Greedy search configuration
    parser.add_argument("--greedy-search", action="store_true", default=False,
                       help="Use greedy search process evaluation (AIME24 and GAIA only)")
    parser.add_argument("--greedy-candidates", type=int, default=3,
                       help="Number of candidates per round for greedy search (default: 3)")
    parser.add_argument("--greedy-rounds", type=int, default=2,
                       help="Number of rounds for greedy search (default: 2)")
    
    # AgentWise greedy search configuration
    parser.add_argument("--agentwise-greedy-search", action="store_true", default=False,
                       help="Use agent-wise greedy search process evaluation (AIME24 and GAIA only)")
    parser.add_argument("--agentwise-greedy-candidates", type=int, default=3,
                       help="Number of candidates per agent for agent-wise greedy search (default: 3)")
    parser.add_argument("--agentwise-greedy-rounds", type=int, default=2,
                       help="Number of rounds for agent-wise greedy search (default: 2)")
    
    # AgentWise beam search configuration
    parser.add_argument("--agentwise-beam-search", action="store_true", default=False,
                       help="Use agent-wise beam search process evaluation (AIME24 and GAIA only)")
    parser.add_argument("--agentwise-beam-width", type=int, default=3,
                       help="Beam width per agent for agent-wise beam search (default: 3)")
    parser.add_argument("--agentwise-beam-rounds", type=int, default=3,
                       help="Number of rounds for agent-wise beam search (default: 2)")
    
    # Judge type configuration
    parser.add_argument("--judge-type", type=str, choices=["scoring", "ranking", "prm"], default="scoring",
                       help="Type of judge to use: 'scoring' (0-5 scale), 'ranking' (relative ordering), or 'prm' (Process Reward Model)")
    
    # PRM configuration (only used when --judge-type prm)
    parser.add_argument("--prm-model-path", type=str, 
                       default="/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B",
                       help="Path to PRM model (default: Qwen2.5-Math-PRM-7B)")
    parser.add_argument("--prm-api-url", type=str, default="http://localhost:8000/pooling",
                       help="PRM API endpoint URL (default: http://localhost:8000/pooling)")
    parser.add_argument("--prm-enable-summarization", action="store_true", default=False,
                       help="Enable summarization for long responses when using PRM")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create debate configuration
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
    print(f"  Benchmark: {args.benchmark}")
    if args.beam_search:
        print(f"  Beam Search: ENABLED")
        print(f"  Beam Width: {args.beam_width}")
        print(f"  Beam Rounds: {args.beam_rounds}")
        print(f"  Judge Type: {args.judge_type}")
        if args.judge_type == "prm":
            print(f"  PRM Model: {args.prm_model_path}")
            print(f"  PRM API: {args.prm_api_url}")
            print(f"  PRM Summarization: {args.prm_enable_summarization}")
    if args.greedy_search:
        print(f"  Greedy Search: ENABLED")
        print(f"  Greedy Candidates: {args.greedy_candidates}")
        print(f"  Greedy Rounds: {args.greedy_rounds}")
        print(f"  Judge Type: {args.judge_type}")
        if args.judge_type == "prm":
            print(f"  PRM Model: {args.prm_model_path}")
            print(f"  PRM API: {args.prm_api_url}")
            print(f"  PRM Summarization: {args.prm_enable_summarization}")
    if args.agentwise_greedy_search:
        print(f"  AgentWise Greedy Search: ENABLED")
        print(f"  AgentWise Greedy Candidates: {args.agentwise_greedy_candidates}")
        print(f"  AgentWise Greedy Rounds: {args.agentwise_greedy_rounds}")
        print(f"  Judge Type: {args.judge_type}")
        if args.judge_type == "prm":
            print(f"  PRM Model: {args.prm_model_path}")
            print(f"  PRM API: {args.prm_api_url}")
            print(f"  PRM Summarization: {args.prm_enable_summarization}")
    if args.agentwise_beam_search:
        print(f"  AgentWise Beam Search: ENABLED")
        print(f"  AgentWise Beam Width: {args.agentwise_beam_width}")
        print(f"  AgentWise Beam Rounds: {args.agentwise_beam_rounds}")
        print(f"  Judge Type: {args.judge_type}")
        if args.judge_type == "prm":
            print(f"  PRM Model: {args.prm_model_path}")
            print(f"  PRM API: {args.prm_api_url}")
            print(f"  PRM Summarization: {args.prm_enable_summarization}")
    print()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # try:
    if args.benchmark in ["gaia", "all"]:
        print(f"\n{'='*60}")
        if args.gaia_use_tools:
            print("RUNNING GAIA EVALUATION (with web search enabled)")
        else:
            print("RUNNING GAIA EVALUATION (without web search)")
        print(f"{'='*60}")
        config.use_tools = args.gaia_use_tools  # Use command-line flag
        
        if args.beam_search:
            results["gaia"] = run_gaia_beam_search(
                config, 
                args.max_gaia_examples,
                args.beam_width,
                args.beam_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        elif args.greedy_search:
            results["gaia"] = run_gaia_greedy_search(
                config, 
                args.max_gaia_examples,
                args.greedy_candidates,
                args.greedy_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        elif args.agentwise_greedy_search:
            results["gaia"] = run_gaia_agentwise_greedy_search(
                config, 
                args.max_gaia_examples,
                args.agentwise_greedy_candidates,
                args.agentwise_greedy_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        elif args.agentwise_beam_search:
            results["gaia"] = run_gaia_agentwise_beam_search(
                config, 
                args.max_gaia_examples,
                args.agentwise_beam_width,
                args.agentwise_beam_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        else:
            results["gaia"] = run_gaia(config, args.max_gaia_examples)
        
        config.use_tools = False  # Reset for other benchmarks
    
    if args.benchmark in ["aime24", "all"]:
        print(f"\n{'='*60}")
        print("RUNNING AIME24 EVALUATION")
        if args.aime_problem_ids:
            print(f"Specific Problem IDs: {args.aime_problem_ids}")
        print(f"{'='*60}")
        
        if args.beam_search:
            results["aime24"] = run_aime24_beam_search(
                config, 
                args.max_aime_examples, 
                args.aime_problem_ids,
                args.beam_width,
                args.beam_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        elif args.greedy_search:
            results["aime24"] = run_aime24_greedy_search(
                config, 
                args.max_aime_examples, 
                args.aime_problem_ids,
                args.greedy_candidates,
                args.greedy_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        elif args.agentwise_greedy_search:
            results["aime24"] = run_aime24_agentwise_greedy_search(
                config, 
                args.max_aime_examples, 
                args.aime_problem_ids,
                args.agentwise_greedy_candidates,
                args.agentwise_greedy_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        elif args.agentwise_beam_search:
            results["aime24"] = run_aime24_agentwise_beam_search(
                config, 
                args.max_aime_examples, 
                args.aime_problem_ids,
                args.agentwise_beam_width,
                args.agentwise_beam_rounds,
                args.judge_type,
                args.prm_model_path,
                args.prm_api_url,
                args.prm_enable_summarization,
                args.output_dir
            )
        else:
            results["aime24"] = run_aime24(config, args.max_aime_examples, args.aime_problem_ids)
    
    if args.benchmark in ["aime25", "all"]:
            print(f"\n{'='*60}")
            print("RUNNING AIME25 EVALUATION")
            if args.aime25_problem_ids:
                print(f"Specific Problem IDs: {args.aime25_problem_ids}")
            print(f"{'='*60}")
            
            if args.beam_search:
                results["aime25"] = _run_search_evaluation(
                    "aime25", "Beam Search PRM", BeamSearchConfig, 
                    "run_beam_search_evaluation", config, args.max_aime25_examples, args.aime25_problem_ids,
                    args.judge_type, args.prm_model_path, args.prm_api_url, args.prm_enable_summarization,
                    args.output_dir,
                    beam_width=args.beam_width, num_rounds=args.beam_rounds
                )
            elif args.greedy_search:
                results["aime25"] = _run_search_evaluation(
                    "aime25", "Greedy Search PRM", GreedySearchConfig,
                    "run_greedy_search_evaluation", config, args.max_aime25_examples, args.aime25_problem_ids,
                    args.judge_type, args.prm_model_path, args.prm_api_url, args.prm_enable_summarization,
                    args.output_dir,
                    num_candidates=args.greedy_candidates, num_rounds=args.greedy_rounds
                )
            elif args.agentwise_greedy_search:
                results["aime25"] = _run_search_evaluation(
                    "aime25", "AgentWise Greedy Search PRM", AgentWiseGreedySearchConfig,
                    "run_agentwise_greedy_search_evaluation", config, args.max_aime25_examples, args.aime25_problem_ids,
                    args.judge_type, args.prm_model_path, args.prm_api_url, args.prm_enable_summarization,
                    args.output_dir,
                    num_candidates=args.agentwise_greedy_candidates, num_rounds=args.agentwise_greedy_rounds
                )
            elif args.agentwise_beam_search:
                results["aime25"] = _run_search_evaluation(
                    "aime25", "AgentWise Beam Search PRM", AgentWiseBeamSearchConfig,
                    "run_agentwise_beam_search_evaluation", config, args.max_aime25_examples, args.aime25_problem_ids,
                    args.judge_type, args.prm_model_path, args.prm_api_url, args.prm_enable_summarization,
                    args.output_dir,
                    beam_width=args.agentwise_beam_width, num_rounds=args.agentwise_beam_rounds
                )
            else:
                results["aime25"] = run_aime25(config, args.max_aime25_examples, args.aime25_problem_ids)
            
    # except KeyboardInterrupt:
    #     print("\nEvaluation interrupted by user")
    # except Exception as e:
    #     print(f"Error during evaluation: {e}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved in: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main()