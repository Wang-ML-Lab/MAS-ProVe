"""
Agent-Wise Beam Search Process Reward Model (PRM) System
Maintains beam width for each agent independently across rounds
"""

import asyncio
import random
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import SearchNode as BeamNode, parse_judge_score, call_judge_llm, parse_ranking, calculate_prm_score
from llm_debate_tool_call import DebateConfig, debate, direct, debate_refine
from prompts.judge_prompts import judge
from prompts.ranking_judge_prompts import judge_ranking
import openai
import backoff


@dataclass
class BeamSearchConfig:
    """Configuration for agent-wise beam search process evaluation"""
    beam_width: int = 3  # Number of beams to maintain per agent
    num_rounds: int = 3  # Total number of refinement rounds (0 = initial, 1+ = refine)
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


class BeamSearchPRM:
    """
    Agent-Wise Beam Search with Process Reward Model.
    
    Process:
    1. Round 0: Generate beam_width (e.g., 3) initial responses for each agent (NO judging)
    2. Refinement rounds: 
       - Each agent's beam response sees ALL other agent's beam responses
       - Agent1 beam[0] paired with Agent2 beam[0,1,2] -> 3 refinements
       - Agent1 beam[1] paired with Agent2 beam[0,1,2] -> 3 refinements  
       - Agent1 beam[2] paired with Agent2 beam[0,1,2] -> 3 refinements
       - Total: 9 refinements per agent
       - Judge selects best beam_width refinements for each agent
    3. Final: Check if ANY of the final beam responses (from both agents) is correct
    """
    
    def __init__(self, config: BeamSearchConfig):
        self.config = config
        self.all_nodes = []
        self.judge_usage = []
    
    async def judge_and_select_top_k(self, question: str, responses: List[str], 
                                      agent_id: int, k: int) -> List[tuple]:
        """
        Judge multiple responses and select top k
        
        Returns:
            List of (response, score, explanation, usage) tuples for top k
        """
        if self.config.judge_type == "ranking":
            ranking, explanation, usage = await judge_and_rank_responses(
                self.config.judge_model, question, responses
            )
            # Track usage
            self.judge_usage.append(usage)
            
            # Select top k from ranking
            top_results = []
            for i in range(min(k, len(ranking))):
                idx = ranking[i]
                score = float(len(responses) - i)  # Higher rank = higher score
                top_results.append((responses[idx], score, explanation, usage))
                print(f"    Agent {agent_id + 1} Beam {i + 1}: Candidate {idx + 1} (rank {i + 1})")
            
            return top_results
            
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
            
            # Track usage
            if usages:
                self.judge_usage.extend(usages)
            
            # Sort by score and take top k
            scored_responses = [(responses[i], scores[i], f"PRM Score: {scores[i]:.4f}", usages) 
                               for i in range(len(responses))]
            scored_responses.sort(key=lambda x: x[1], reverse=True)
            
            top_results = scored_responses[:k]
            for i, (_, score, _, _) in enumerate(top_results):
                print(f"    Agent {agent_id + 1} Beam {i + 1}: Score {score:.4f} (PRM)")
            
            return top_results
            
        else:  # scoring
            # Judge all responses
            judge_tasks = [
                judge_response(self.config.judge_model, question, resp)
                for resp in responses
            ]
            results = await asyncio.gather(*judge_tasks)
            
            # Track usage
            for r in results:
                self.judge_usage.append(r[2])
            
            # Sort by score and take top k
            scored_responses = [(responses[i], results[i][0], results[i][1], results[i][2]) 
                               for i in range(len(responses))]
            scored_responses.sort(key=lambda x: x[1], reverse=True)
            
            top_results = scored_responses[:k]
            for i, (_, score, _, _) in enumerate(top_results):
                print(f"    Agent {agent_id + 1} Beam {i + 1}: Score {score:.2f}")
            
            return top_results
    
    async def search(self, question: str) -> Dict[str, Any]:
        """
        Run agent-wise beam search with process evaluation.
        
        Args:
            question: The problem to solve
            
        Returns:
            Dictionary with search results
        """
        full_question = question
        judge_type_display = self.config.judge_type.upper()
        
        print(f"\n{'='*60}")
        print(f"AGENT-WISE BEAM SEARCH ({judge_type_display})")
        print(f"{'='*60}")
        print(f"Beam Width: {self.config.beam_width}")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Agents: {self.config.debate_config.num_agents}")
        
        num_agents = self.config.debate_config.num_agents
        
        # Track beam for each agent (list of nodes)
        agent_beams = {i: [] for i in range(num_agents)}  # agent_id -> list of nodes
        
        # ============== ROUND 0: Initial Generation (NO JUDGING) ==============
        print(f"\n{'='*60}")
        print(f"ROUND 0: Initial Generation (No Judging)")
        print(f"{'='*60}")
        
        # Generate beam_width initial responses for each agent
        print(f"\n  Generating {self.config.beam_width} initial responses for each of {num_agents} agents (in parallel)...")
        
        all_initial_tasks = []
        for agent_id in range(num_agents):
            for beam_idx in range(self.config.beam_width):
                all_initial_tasks.append((
                    agent_id,
                    beam_idx,
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
        all_results = await asyncio.gather(*[task for _, _, task in all_initial_tasks])
        
        # Create nodes for all initial responses (no judging)
        for idx, (agent_id, beam_idx, _) in enumerate(all_initial_tasks):
            response, usage = all_results[idx]
            
            node = BeamNode(
                response=response,
                score=0.0,  # No score yet (not judged)
                judge_explanation="Initial generation (not judged)",
                round=0,
                parent=None,
                trajectory=[response],
                usage=[usage] if usage else []
            )
            
            agent_beams[agent_id].append(node)
            self.all_nodes.append(node)
            print(f"  ✓ Agent {agent_id + 1} Beam {beam_idx + 1}: Generated")
        
        print(f"\n  ✓ Round 0 complete: {self.config.beam_width} beams per agent (not judged)")
        
        # ============== REFINEMENT ROUNDS ==============
        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}: Refinement with Cross-Beam Pairing")
            print(f"{'='*60}")
            
            # For each agent, generate refinements
            for agent_id in range(num_agents):
                # Get other agents' indices
                other_agent_ids = [i for i in range(num_agents) if i != agent_id]
                
                print(f"\n  Agent {agent_id + 1}: Generating refinements...")
                print(f"    Current beam size: {len(agent_beams[agent_id])}")
                
                # Generate all refinement combinations
                all_refine_tasks = []
                
                # For each beam in current agent
                for beam_node in agent_beams[agent_id]:
                    # For each combination of other agents' beams
                    # If 2 agents, other_agent has beam_width beams
                    # We pair this beam with each of those
                    
                    if num_agents == 2:
                        # Simple case: pair with each beam from other agent
                        other_agent_id = other_agent_ids[0]
                        other_beams = agent_beams[other_agent_id]
                        
                        for other_beam_node in other_beams:
                            all_refine_tasks.append((
                                beam_node,
                                debate_refine(
                                    model=self.config.debate_config.model,
                                    question=full_question,
                                    original_cot_response=beam_node.response,
                                    other_agents_responses=[other_beam_node.response],
                                    dataset=self.config.debate_config.dataset,
                                    use_tools=self.config.debate_config.use_tools,
                                    temperature=self.config.debate_config.temperature,
                                    max_tokens=self.config.debate_config.max_tokens
                                )
                            ))
                    else:
                        # Multi-agent case: use one beam from each other agent
                        # For simplicity, pair with first beam of each other agent
                        # (This can be extended to all combinations if needed)
                        other_responses = [agent_beams[oid][0].response for oid in other_agent_ids]
                        
                        all_refine_tasks.append((
                            beam_node,
                            debate_refine(
                                model=self.config.debate_config.model,
                                question=full_question,
                                original_cot_response=beam_node.response,
                                other_agents_responses=other_responses,
                                dataset=self.config.debate_config.dataset,
                                use_tools=self.config.debate_config.use_tools,
                                temperature=self.config.debate_config.temperature,
                                max_tokens=self.config.debate_config.max_tokens
                            )
                        ))
                
                print(f"    Generating {len(all_refine_tasks)} refinements (in parallel)...")
                
                # Execute all refinement tasks in parallel
                all_results = await asyncio.gather(*[task for _, task in all_refine_tasks])
                
                # Collect all refined responses
                refined_responses = []
                refined_usages = []
                parent_nodes = []
                
                for idx, (parent_node, _) in enumerate(all_refine_tasks):
                    response, usage = all_results[idx]
                    refined_responses.append(response)
                    refined_usages.append(usage)
                    parent_nodes.append(parent_node)
                
                print(f"    Generated {len(refined_responses)} refined responses")
                print(f"    Judging and selecting top {self.config.beam_width}...")
                
                # Judge and select top beam_width
                top_results = await self.judge_and_select_top_k(
                    full_question, refined_responses, agent_id, self.config.beam_width
                )
                
                # Create nodes for top results and update beam
                new_beam = []
                for response, score, explanation, judge_usage in top_results:
                    # Find which parent this came from
                    resp_idx = refined_responses.index(response)
                    parent_node = parent_nodes[resp_idx]
                    usage = refined_usages[resp_idx]
                    
                    node = BeamNode(
                        response=response,
                        score=score,
                        judge_explanation=explanation,
                        round=round_num,
                        parent=parent_node,
                        trajectory=parent_node.get_full_trajectory() + [response],
                        usage=[usage] if usage else []
                    )
                    
                    new_beam.append(node)
                    self.all_nodes.append(node)
                
                agent_beams[agent_id] = new_beam
                print(f"    ✓ Agent {agent_id + 1}: Selected top {len(new_beam)} responses")
            
            print(f"\n  ✓ Round {round_num} complete: Updated beams for all agents")
        
        # ============== FINAL: Collect all beam responses ==============
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: Collecting all beam responses")
        print(f"{'='*60}")
        
        # Collect all final beam responses from all agents
        all_final_responses = []
        all_final_nodes = []
        
        for agent_id in range(num_agents):
            print(f"\n  Agent {agent_id + 1} final beams:")
            for beam_idx, node in enumerate(agent_beams[agent_id]):
                all_final_responses.append(node.response)
                all_final_nodes.append(node)
                if self.config.judge_type == "prm":
                    print(f"    Beam {beam_idx + 1}: Score {node.score:.4f} (PRM)")
                elif self.config.judge_type == "ranking":
                    print(f"    Beam {beam_idx + 1}: Score {node.score:.1f}")
                else:
                    print(f"    Beam {beam_idx + 1}: Score {node.score:.2f}/5")
        
        # Find best scoring response overall (for reporting)
        best_node = max(all_final_nodes, key=lambda n: n.score)
        best_agent_id = None
        for agent_id in range(num_agents):
            if best_node in agent_beams[agent_id]:
                best_agent_id = agent_id
                break
        
        print(f"\n{'='*60}")
        print(f"SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Total final responses: {len(all_final_responses)} ({num_agents} agents × {self.config.beam_width} beams)")
        print(f"Best scoring response: Agent {best_agent_id + 1}")
        if self.config.judge_type == "prm":
            print(f"Best Score: {best_node.score:.4f} (PRM)")
        elif self.config.judge_type == "ranking":
            print(f"Best Score: {best_node.score:.1f}")
        else:
            print(f"Best Score: {best_node.score:.2f}/5")
        print(f"Total Nodes Explored: {len(self.all_nodes)}")
        
        return {
            'question': question,
            'best_node': best_node,
            'best_response': best_node.response,
            'best_score': best_node.score,
            'best_trajectory': best_node.get_full_trajectory(),
            'best_agent_id': best_agent_id,
            'all_final_responses': all_final_responses,  # All beam responses from all agents
            'all_final_nodes': all_final_nodes,  # All final beam nodes
            'agent_beams': agent_beams,  # Final beam state per agent
            'all_nodes': self.all_nodes,
            'total_nodes_explored': len(self.all_nodes),
            'judge_usage': self.judge_usage,
            'config': {
                'beam_width': self.config.beam_width,
                'num_rounds': self.config.num_rounds,
                'debate_config': self.config.debate_config.__dict__,
                'judge_model': self.config.judge_model,
                'judge_type': self.config.judge_type
            }
        }
