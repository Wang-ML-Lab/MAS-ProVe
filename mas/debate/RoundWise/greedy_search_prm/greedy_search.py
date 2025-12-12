"""
Greedy Search Process Reward Model (PRM) System
Runs multi-round debate with greedy selection and judge-based process evaluation
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
    Greedy Search Process Reward Model
    
    Supports both scoring and ranking-based judges via judge_type config parameter.
    
    Runs debate in a greedy manner:
    1. Start with num_candidates parallel debate instances (Round 0)
    2. Judge all responses (scoring or ranking)
    3. Select ONLY the best candidate (greedy selection)
    4. Generate num_candidates children from the best node
    5. Judge and select ONLY the best child
    6. Repeat until num_rounds completed
    """
    
    def __init__(self, config: GreedySearchConfig):
        self.config = config
        self.all_nodes: List[GreedyNode] = []
        self.judge_usage: List[Dict] = []
        # Store agent responses for each branch (for multi-agent debate within each branch)
        self.branch_agent_responses: Dict[int, List[List[str]]] = {}  # branch_id -> [round_0_responses, round_1_responses, ...]
        self.node_to_branch: Dict[int, int] = {}  # Map node index to its branch_id
        self.node_counter: int = 0  # Counter for assigning unique IDs to nodes
    
    async def run_initial_round(self, question: str) -> List[GreedyNode]:
        """
        Run initial round: Generate num_candidates branches, each with num_agents initial responses
        All branches run in parallel
        
        Returns:
            List of GreedyNode objects for round 0 (one per branch)
        """
        print(f"\n{'='*60}")
        print(f"ROUND 0: Initial Generation ({self.config.num_candidates} candidates × {self.config.debate_config.num_agents} agents)")
        print(f"{'='*60}")
        
        async def create_branch(branch_id: int):
            """Create a single branch with num_agents initial responses"""
            print(f"  Candidate {branch_id+1}/{self.config.num_candidates}: Generating {self.config.debate_config.num_agents} initial responses...")
            
            # All agents generate initial responses (like first round of debate)
            first_round_tasks = [
                direct(
                    model=self.config.debate_config.model,
                    question=question,
                    dataset=self.config.debate_config.dataset,
                    use_tools=self.config.debate_config.use_tools,
                    temperature=self.config.debate_config.temperature,
                    max_tokens=self.config.debate_config.max_tokens
                )
                for _ in range(self.config.debate_config.num_agents)
            ]
            
            first_round_results = await asyncio.gather(*first_round_tasks)
            first_round_cot_responses = [result[0] for result in first_round_results]
            first_round_usage = [result[1] for result in first_round_results]
            
            # Store agent responses for this branch
            self.branch_agent_responses[branch_id] = [first_round_cot_responses]
            
            # Randomly pick one response from this branch to represent it
            branch_response = random.choice(first_round_cot_responses)
            
            # Create node for this branch
            node_id = self.node_counter
            self.node_counter += 1
            
            node = GreedyNode(
                response=branch_response,
                score=0.0,  # Will be set by judge
                round=0,
                parent=None,
                trajectory=[branch_response],
                usage=[u for u in first_round_usage if u is not None]
            )
            self.node_to_branch[node_id] = branch_id  # Track which branch this node belongs to
            print(f"  ✓ Candidate {branch_id+1} complete: {len(branch_response)} chars")
            
            return node
        
        # Run all branches in parallel
        branch_tasks = [create_branch(branch_id) for branch_id in range(self.config.num_candidates)]
        nodes = await asyncio.gather(*branch_tasks)
        
        # Add all nodes to all_nodes
        for node in nodes:
            self.all_nodes.append(node)
        
        return nodes
    
    async def judge_nodes(self, question: str, nodes: List[GreedyNode]) -> List[GreedyNode]:
        """
        Judge all nodes using either scoring, ranking, or PRM based on config
        
        Args:
            question: Original question
            nodes: List of GreedyNode to judge
        
        Returns:
            Same nodes with updated scores
        """
        print(f"\n⚖️  Judging {len(nodes)} responses ({self.config.judge_type})...")
        
        if self.config.judge_type == "prm":
            # Use PRM to score all responses
            responses = [node.response for node in nodes]
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
            
            for node, score in zip(nodes, scores):
                node.score = score
                node.judge_explanation = f"PRM Score: {score:.4f}"
            
            # Track summarization usage
            if usages:
                self.judge_usage.extend(usages)
            
            for i, node in enumerate(nodes, 1):
                print(f"  Response {i}: {node.score:.4f} (PRM)")
        
        elif self.config.judge_type == "ranking":
            # Ranking-based judging
            responses = [node.response for node in nodes]
            ranking, explanation, usage = await judge_and_rank_responses(
                self.config.judge_model, question, responses
            )
            
            if usage:
                self.judge_usage.append(usage)
            
            # Assign scores based on ranking (best = highest score)
            num_candidates = len(nodes)
            for rank_position, node_idx in enumerate(ranking):
                score = float(num_candidates - rank_position)  # Best gets num_candidates, worst gets 1
                nodes[node_idx].score = score
                nodes[node_idx].judge_explanation = f"Rank: {rank_position + 1}/{num_candidates}\n{explanation}"
                print(f"    Node {node_idx + 1}: Rank {rank_position + 1}/{num_candidates} (Score: {score:.1f})")
        else:
            # Scoring-based judging (default)
            judge_tasks = [
                judge_response(self.config.judge_model, question, node.response)
                for node in nodes
            ]
            
            judge_results = await asyncio.gather(*judge_tasks)
            
            for node, (score, explanation, usage) in zip(nodes, judge_results):
                node.score = score
                node.judge_explanation = explanation
                if usage:
                    self.judge_usage.append(usage)
                print(f"    Node (round {node.round}): Score {score:.2f}/5")
        
        return nodes
    
    def select_best(self, nodes: List[GreedyNode]) -> GreedyNode:
        """
        Select the single best node based on judge scores (greedy selection)
        
        Args:
            nodes: List of candidate nodes
        
        Returns:
            Best scoring node
        """
        best_node = max(nodes, key=lambda n: n.score)
        
        print(f"\n📊 Selected best candidate (GREEDY - {self.config.judge_type.upper()}):")
        if self.config.judge_type == "ranking":
            print(f"  Score: {best_node.score:.1f} (rank-based)")
        elif self.config.judge_type == "prm":
            print(f"  Score: {best_node.score:.4f} (PRM)")
        else:
            print(f"  Score: {best_node.score:.2f}/5")
        
        return best_node
    
    async def refine_branch(self, question: str, parent_node: GreedyNode, parent_node_id: int, round_num: int) -> List[GreedyNode]:
        """
        Refine a single branch by running num_candidates refinement iterations in parallel
        Each iteration runs a full agent debate internally, creating num_candidates child branches
        
        Args:
            question: Original question
            parent_node: Parent node representing this branch
            parent_node_id: ID of the parent node
            round_num: Current round number
        
        Returns:
            List of num_candidates new GreedyNode candidates from this branch
        """
        # Get the branch_id from parent
        parent_branch_id = self.node_to_branch[parent_node_id]
        
        # Get all previous agent responses for this branch
        parent_all_round_responses = self.branch_agent_responses[parent_branch_id]
        
        async def create_child(child_idx: int):
            """Create a single child branch by running refinement"""
            # Copy the agent responses history for the new branch
            all_round_responses = [list(round_resps) for round_resps in parent_all_round_responses]
            
            # Run refinement for each agent (they see each other's responses within the branch)
            refine_tasks = [
                debate_refine(
                    model=self.config.debate_config.model,
                    question=question,
                    original_cot_response=current_cot,
                    other_agents_responses=all_round_responses[-1][:i] + all_round_responses[-1][i+1:],
                    dataset=self.config.debate_config.dataset,
                    use_tools=self.config.debate_config.use_tools,
                    temperature=self.config.debate_config.temperature,
                    max_tokens=self.config.debate_config.max_tokens
                )
                for i, current_cot in enumerate(all_round_responses[-1])
            ]
            
            refine_results = await asyncio.gather(*refine_tasks)
            refine_responses = [result[0] for result in refine_results]
            refine_usage = [result[1] for result in refine_results]
            
            # Update branch agent responses
            all_round_responses.append(refine_responses)
            
            # Randomly pick one response to represent this child branch
            branch_response = random.choice(refine_responses)
            
            return all_round_responses, branch_response, refine_usage
        
        # Run all child creations in parallel
        child_tasks = [create_child(i) for i in range(self.config.num_candidates)]
        child_results = await asyncio.gather(*child_tasks)
        
        # Create nodes for all children
        child_nodes = []
        for all_round_responses, branch_response, refine_usage in child_results:
            # Create a new branch_id for each child
            child_branch_id = len(self.branch_agent_responses)
            
            # Store this child's branch history
            self.branch_agent_responses[child_branch_id] = all_round_responses
            
            # Assign unique ID to child node
            child_node_id = self.node_counter
            self.node_counter += 1
            
            child_node = GreedyNode(
                response=branch_response,
                score=0.0,  # Will be set by judge
                round=round_num,
                parent=parent_node,
                trajectory=parent_node.get_full_trajectory() + [branch_response],
                usage=[u for u in refine_usage if u is not None]
            )
            child_nodes.append(child_node)
            self.all_nodes.append(child_node)
            self.node_to_branch[child_node_id] = child_branch_id
        
        return child_nodes
    
    async def run_refinement_round(self, question: str, parent_node: GreedyNode, round_num: int) -> GreedyNode:
        """
        Run one refinement round: expand the parent node into num_candidates children
        Judge all children and select ONLY the best one (greedy)
        
        Args:
            question: Original question
            parent_node: Parent node to expand from
            round_num: Current round number
        
        Returns:
            Best child node selected by judge
        """
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}: Refinement ({self.config.num_candidates} candidates)")
        print(f"  Each candidate: {self.config.debate_config.num_agents} agents refine internally")
        print(f"{'='*60}")
        
        # Find parent node ID
        parent_node_id = self.all_nodes.index(parent_node)
        
        print(f"  Expanding from best node (score: {parent_node.score:.2f})...")
        
        # Generate num_candidates children from the parent
        child_nodes = await self.refine_branch(question, parent_node, parent_node_id, round_num)
        
        print(f"  ✓ Generated {len(child_nodes)} candidate branches")
        
        # Judge all children
        child_nodes = await self.judge_nodes(question, child_nodes)
        
        # Select ONLY the best child (greedy)
        best_child = self.select_best(child_nodes)
        
        return best_child
    
    async def search(self, question: str) -> Dict[str, Any]:
        """
        Run complete greedy search process evaluation
        
        Args:
            question: Question to solve
        
        Returns:
            Dictionary with results including best trajectory
        """
        full_question = question
        
        print(f"\n{'#'*60}")
        print(f"GREEDY SEARCH PROCESS EVALUATION ({self.config.judge_type.upper()})")
        print(f"{'#'*60}")
        print(f"Question: {question[:100]}...")
        print(f"Candidates per round: {self.config.num_candidates}")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Model: {self.config.debate_config.model}")
        print(f"Judge: {self.config.judge_model}")
        
        # Round 0: Initial generation
        candidate_nodes = await self.run_initial_round(full_question)
        candidate_nodes = await self.judge_nodes(full_question, candidate_nodes)
        
        # Select ONLY the best candidate (greedy)
        best_node = self.select_best(candidate_nodes)
        
        # Refinement rounds
        for round_num in range(1, self.config.num_rounds + 1):
            # Expand the best node and select best child
            best_node = await self.run_refinement_round(full_question, best_node, round_num)
        
        print(f"\n{'='*60}")
        print(f"SEARCH COMPLETE")
        print(f"{'='*60}")
        if self.config.judge_type == "ranking":
            print(f"Final Score: {best_node.score:.1f} (Rank-based)")
        else:
            print(f"Final Score: {best_node.score:.2f}/5 (Score-based)")
        print(f"Final Node Round: {best_node.round}")
        print(f"Trajectory Length: {len(best_node.get_full_trajectory())}")
        
        return {
            "question": question,
            "best_node": best_node,
            "best_score": best_node.score,
            "best_trajectory": best_node.get_full_trajectory(),
            "best_response": best_node.response,
            "all_nodes": self.all_nodes,
            "final_node": best_node,
            "config": {
                "num_candidates": self.config.num_candidates,
                "num_rounds": self.config.num_rounds,
                "debate_config": self.config.debate_config.__dict__,
                "judge_model": self.config.judge_model
            },
            "total_nodes_explored": len(self.all_nodes),
            "judge_usage": self.judge_usage
        }
