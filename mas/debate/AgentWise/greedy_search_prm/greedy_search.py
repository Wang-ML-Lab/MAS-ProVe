"""
Agent-Wise Greedy Search Process Reward Model (PRM) System
Evaluates each agent independently, selecting best response per agent (not per branch)
"""

import asyncio
import random
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import SearchNode as GreedyNode, parse_judge_score, call_judge_llm, parse_ranking, calculate_prm_score
# from llm_debate import DebateConfig, direct, debate_refine
from llm_debate_tool_call import DebateConfig, direct, debate_refine
from prompts.judge_prompts import judge
from prompts.ranking_judge_prompts import judge_ranking
import openai
import backoff


@dataclass
class GreedySearchConfig:
    """Configuration for greedy search process evaluation"""
    num_candidates: int = 3  # Number of parallel candidates to generate at each step
    num_rounds: int = 3  # Total number of refinement rounds
    debate_config: DebateConfig = None  # Configuration for underlying debate
    judge_model: str = "gpt-5-mini"  # Model for judge evaluation
    judge_type: str = "scoring"  # "scoring", "ranking", or "prm"
    prm_model_path: str = "/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B"  # Path to PRM model
    prm_api_url: str = "http://localhost:8000/pooling"  # PRM API endpoint
    enable_summarization: bool = False  # Whether to use summarization for long responses
    
    def __post_init__(self):
        if self.debate_config is None:
            self.debate_config = DebateConfig()


async def judge_response(model: str, question: str, response: str) -> tuple:
    """
    Judge a single response using the judge model
    
    Args:
        model: Model name for the judge
        question: Original question
        response: Response to judge
    
    Returns:
        Tuple of (score, explanation, usage)
    """
    messages = judge(model, question, response)
    judge_output, usage = await call_judge_llm(model, messages)
    score = parse_judge_score(judge_output)
    
    return score, judge_output, usage


async def judge_and_rank_responses(model: str, question: str, responses: List[str]) -> tuple:
    """
    Judge and rank multiple responses using the judge model
    
    Args:
        model: Model name for the judge
        question: Original question
        responses: List of responses to rank
    
    Returns:
        Tuple of (ranking, explanation, usage)
        ranking is a list of indices from best to worst (0-indexed)
    """
    messages = judge_ranking(model, question, responses)
    judge_output, usage = await call_judge_llm(model, messages)
    ranking = parse_ranking(judge_output, len(responses))
    
    return ranking, judge_output, usage

class GreedySearchPRM:
    """
    Agent-Wise Greedy Search with Process Reward Model.
    
    Process:
    1. Round 0: Generate 3 candidates for each agent (in parallel), judge selects best per agent
    2. Refinement rounds: For each agent, generate 3 refinements (seeing other agent), select best
    3. Final: Judge selects best response between the two agents
    """
    
    def __init__(self, config: GreedySearchConfig):
        self.config = config
        self.all_nodes = []
        self.judge_usage = []
    
    async def judge_agent_candidates(self, question: str, responses: List[str], agent_id: int) -> tuple:
        """Judge multiple candidate responses and select best"""
        if self.config.judge_type == "ranking":
            ranking, explanation, usage = await judge_and_rank_responses(
                self.config.judge_model, question, responses
            )
            best_idx = ranking[0]  # First in ranking is best
            best_response = responses[best_idx]
            best_score = float(len(responses))  # Best gets highest score
            print(f"    Agent {agent_id + 1}: Candidate {best_idx + 1} ranked best")
        elif self.config.judge_type == "prm":
            # Use PRM to score all responses
            prm_tasks = [
                calculate_prm_score(
                    prompt=question,
                    final_response=resp,
                    prm_model_path=self.config.prm_model_path,
                    api_url=self.config.prm_api_url,
                    enable_summarization=self.config.enable_summarization
                )
                for resp in responses
            ]
            results = await asyncio.gather(*prm_tasks)
            scores = [r[0] for r in results]
            usages = [r[1] for r in results if r[1]]  # Collect non-empty usage dicts
            
            best_idx = scores.index(max(scores))
            best_response = responses[best_idx]
            best_score = scores[best_idx]
            explanation = f"PRM Score: {best_score:.4f}"
            usage = usages  # Return list of summarization usages
            print(f"    Agent {agent_id + 1}: Candidate {best_idx + 1} scored {best_score:.4f} (PRM)")
        else:  # scoring
            # Judge all responses and select highest score
            judge_tasks = [
                judge_response(self.config.judge_model, question, resp)
                for resp in responses
            ]
            results = await asyncio.gather(*judge_tasks)
            scores = [r[0] for r in results]
            explanations = [r[1] for r in results]
            usages = [r[2] for r in results]
            
            best_idx = scores.index(max(scores))
            best_response = responses[best_idx]
            best_score = scores[best_idx]
            explanation = explanations[best_idx]
            usage = usages
            print(f"    Agent {agent_id + 1}: Candidate {best_idx + 1} scored {best_score:.2f}")
        
        return best_response, best_score, explanation, usage
    
    async def search(self, question: str) -> Dict[str, Any]:
        """
        Run agent-wise greedy search with process evaluation.
        
        Args:
            question: The problem to solve
            
        Returns:
            Dictionary with search results
        """
        full_question = question
        judge_type_display = self.config.judge_type.upper()
        
        print(f"\n{'='*60}")
        print(f"AGENT-WISE GREEDY SEARCH ({judge_type_display})")
        print(f"{'='*60}")
        print(f"Candidates per agent: {self.config.num_candidates}")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Agents: {self.config.debate_config.num_agents}")
        
        num_agents = self.config.debate_config.num_agents
        
        # Track best response for each agent
        agent_nodes = [None] * num_agents  # Current best node for each agent
        
        # ============== ROUND 0: Initial Generation ==============
        print(f"\n{'='*60}")
        print(f"ROUND 0: Initial Generation")
        print(f"{'='*60}")
        
        # Generate all candidates for all agents in parallel
        print(f"\n  Generating {self.config.num_candidates} candidates for each of {num_agents} agents (in parallel)...")
        
        all_initial_tasks = []
        for agent_id in range(num_agents):
            # Generate num_candidates initial responses for this agent
            for _ in range(self.config.num_candidates):
                all_initial_tasks.append((
                    agent_id,
                    direct(
                        model=self.config.debate_config.model,
                        question=full_question,
                        dataset=self.config.debate_config.dataset,
                        use_tools=self.config.debate_config.use_tools,
                        temperature=self.config.debate_config.temperature,
                        max_tokens=self.config.debate_config.max_tokens
                    )
                ))
        
        # Execute all generation tasks in parallel
        all_results = await asyncio.gather(*[task for _, task in all_initial_tasks])
        
        # Group results by agent
        agent_responses = {i: [] for i in range(num_agents)}
        agent_usages = {i: [] for i in range(num_agents)}
        
        for idx, (agent_id, _) in enumerate(all_initial_tasks):
            response, usage = all_results[idx]
            agent_responses[agent_id].append(response)
            agent_usages[agent_id].append(usage)
        
        # Judge each agent's candidates
        print(f"  Judging candidates for each agent...")
        for agent_id in range(num_agents):
            responses = agent_responses[agent_id]
            usages = agent_usages[agent_id]
            
            best_response, best_score, explanation, judge_usage = await self.judge_agent_candidates(
                full_question, responses, agent_id
            )
            
            # Track usage
            if isinstance(judge_usage, list):
                self.judge_usage.extend(judge_usage)
            else:
                self.judge_usage.append(judge_usage)
            
            # Create node for this agent's best response
            node = GreedyNode(
                response=best_response,
                score=best_score,
                judge_explanation=explanation,
                round=0,
                parent=None,
                trajectory=[best_response],
                usage=[u for u in usages if u is not None]
            )
            
            self.all_nodes.append(node)
            agent_nodes[agent_id] = node
        
        print(f"\n  ✓ Round 0 complete: Selected best response for each agent")
        
        # ============== REFINEMENT ROUNDS ==============
        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}: Refinement")
            print(f"{'='*60}")
            
            # Get current best responses for all agents
            current_agent_responses = [agent_nodes[i].response for i in range(num_agents)]
            
            # Generate all refinements for all agents in parallel
            print(f"\n  Generating {self.config.num_candidates} refinements for each of {num_agents} agents (in parallel)...")
            
            all_refine_tasks = []
            for agent_id in range(num_agents):
                # Get other agents' responses (for this agent to see)
                other_agent_responses = current_agent_responses[:agent_id] + current_agent_responses[agent_id+1:]
                
                # Generate num_candidates refinements for this agent
                for _ in range(self.config.num_candidates):
                    all_refine_tasks.append((
                        agent_id,
                        debate_refine(
                            model=self.config.debate_config.model,
                            question=full_question,
                            original_cot_response=agent_nodes[agent_id].response,
                            other_agents_responses=other_agent_responses,
                            dataset=self.config.debate_config.dataset,
                            use_tools=self.config.debate_config.use_tools,
                            temperature=self.config.debate_config.temperature,
                            max_tokens=self.config.debate_config.max_tokens
                        )
                    ))
            
            # Execute all refinement tasks in parallel
            all_results = await asyncio.gather(*[task for _, task in all_refine_tasks])
            
            # Group results by agent
            agent_refined_responses = {i: [] for i in range(num_agents)}
            agent_refine_usages = {i: [] for i in range(num_agents)}
            
            for idx, (agent_id, _) in enumerate(all_refine_tasks):
                response, usage = all_results[idx]
                agent_refined_responses[agent_id].append(response)
                agent_refine_usages[agent_id].append(usage)
            
            # Judge each agent's refinements
            print(f"  Judging refinements for each agent...")
            for agent_id in range(num_agents):
                refined_responses = agent_refined_responses[agent_id]
                refine_usages = agent_refine_usages[agent_id]
                
                best_response, best_score, explanation, judge_usage = await self.judge_agent_candidates(
                    full_question, refined_responses, agent_id
                )
                
                # Track usage
                if isinstance(judge_usage, list):
                    self.judge_usage.extend(judge_usage)
                else:
                    self.judge_usage.append(judge_usage)
                
                # Create node for this agent's best refinement
                parent_node = agent_nodes[agent_id]
                node = GreedyNode(
                    response=best_response,
                    score=best_score,
                    judge_explanation=explanation,
                    round=round_num,
                    parent=parent_node,
                    trajectory=parent_node.get_full_trajectory() + [best_response],
                    usage=[u for u in refine_usages if u is not None]
                )
                
                self.all_nodes.append(node)
                agent_nodes[agent_id] = node
            
            print(f"\n  ✓ Round {round_num} complete: Updated best response for each agent")
        
        # ============== FINAL SELECTION ==============
        print(f"\n{'='*60}")
        print(f"FINAL SELECTION: Judging between {num_agents} agents")
        print(f"{'='*60}")
        
        final_responses = [agent_nodes[i].response for i in range(num_agents)]
        
        if self.config.judge_type == "ranking":
            ranking, explanation, usage = await judge_and_rank_responses(
                self.config.judge_model, full_question, final_responses
            )
            best_agent_id = ranking[0]
            print(f"  Agent {best_agent_id + 1} ranked best overall")
        elif self.config.judge_type == "prm":
            # Use PRM to score final responses
            prm_tasks = [
                calculate_prm_score(
                    prompt=full_question,
                    final_response=resp,
                    prm_model_path=self.config.prm_model_path,
                    api_url=self.config.prm_api_url,
                    enable_summarization=self.config.enable_summarization
                )
                for resp in final_responses
            ]
            results = await asyncio.gather(*prm_tasks)
            scores = [r[0] for r in results]
            usages = [r[1] for r in results if r[1]]  # Collect non-empty usage dicts
            
            best_agent_id = scores.index(max(scores))
            usage = usages  # Return list of summarization usages
            print(f"  Agent {best_agent_id + 1} scored highest: {scores[best_agent_id]:.4f} (PRM)")
        else:  # scoring
            judge_tasks = [
                judge_response(self.config.judge_model, full_question, resp)
                for resp in final_responses
            ]
            results = await asyncio.gather(*judge_tasks)
            scores = [r[0] for r in results]
            
            best_agent_id = scores.index(max(scores))
            usage = [r[2] for r in results]
            print(f"  Agent {best_agent_id + 1} scored highest: {scores[best_agent_id]:.2f}")
        
        if isinstance(usage, list):
            self.judge_usage.extend(usage)
        else:
            self.judge_usage.append(usage)
        
        best_node = agent_nodes[best_agent_id]
        
        print(f"\n{'='*60}")
        print(f"SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Best Agent: Agent {best_agent_id + 1}")
        if self.config.judge_type == "ranking":
            print(f"Final Score: {best_node.score:.1f}")
        elif self.config.judge_type == "prm":
            print(f"Final Score: {best_node.score:.4f} (PRM)")
        else:
            print(f"Final Score: {best_node.score:.2f}/5")
        print(f"Trajectory Length: {len(best_node.get_full_trajectory())}")
        print(f"Total Nodes Explored: {len(self.all_nodes)}")
        
        return {
            'question': question,
            'best_node': best_node,
            'best_response': best_node.response,
            'best_score': best_node.score,
            'best_trajectory': best_node.get_full_trajectory(),
            'best_agent_id': best_agent_id,
            'all_nodes': self.all_nodes,
            'agent_nodes': agent_nodes,  # Final state of all agents
            'total_nodes_explored': len(self.all_nodes),
            'judge_usage': self.judge_usage,
            'config': {
                'num_candidates': self.config.num_candidates,
                'num_rounds': self.config.num_rounds,
                'debate_config': self.config.debate_config.__dict__,
                'judge_model': self.config.judge_model,
                'judge_type': self.config.judge_type
            }
        }
