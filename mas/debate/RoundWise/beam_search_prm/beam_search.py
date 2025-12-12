"""
Beam Search Process Reward Model (PRM) System
Runs multi-round debate with beam search and judge-based process evaluation
"""

import asyncio
import random
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import SearchNode as BeamNode, parse_judge_score, parse_ranking, call_judge_llm, calculate_prm_score
# from llm_debate import DebateConfig, debate, direct, debate_refine
from llm_debate_tool_call import DebateConfig, direct, debate_refine
from prompts.judge_prompts import judge
from prompts.ranking_judge_prompts import judge_ranking
import openai
import backoff


@dataclass
class BeamSearchConfig:
    """Configuration for beam search process evaluation"""
    beam_width: int = 3  # Number of candidates to keep at each round
    num_rounds: int = 3  # Total number of refinement rounds
    debate_config: DebateConfig = None  # Configuration for underlying debate
    judge_model: str = "gpt-5-mini"  # Model for judge evaluation
    select_best: bool = True  # If True, select best scored; if False, sample by score
    judge_type: str = "scoring"  # "scoring" (0-5 scale), "ranking" (relative ordering), or "prm"
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
    Beam Search Process Reward Model
    
    Runs debate in a beam search manner:
    1. Start with beam_width parallel debate instances (Round 0)
    2. Judge all responses
    3. Select top beam_width candidates
    4. Continue each for next round
    5. Repeat until num_rounds completed
    """
    
    def __init__(self, config: BeamSearchConfig):
        self.config = config
        self.all_nodes: List[BeamNode] = []
        self.judge_usage: List[Dict] = []
        # Store agent responses for each branch (for multi-agent debate within each branch)
        self.branch_agent_responses: Dict[int, List[List[str]]] = {}  # branch_id -> [round_0_responses, round_1_responses, ...]
        self.node_to_branch: Dict[int, int] = {}  # Map node index to its branch_id
        self.node_counter: int = 0  # Counter for assigning unique IDs to nodes
    
    async def run_initial_round(self, question: str) -> List[BeamNode]:
        """
        Run initial round: Generate beam_width branches, each with num_agents initial responses
        All branches run in parallel
        
        Returns:
            List of BeamNode objects for round 0 (one per branch)
        """
        print(f"\n{'='*60}")
        print(f"ROUND 0: Initial Generation ({self.config.beam_width} branches × {self.config.debate_config.num_agents} agents)")
        print(f"{'='*60}")
        
        async def create_branch(branch_id: int):
            """Create a single branch with num_agents initial responses"""
            print(f"  Branch {branch_id+1}/{self.config.beam_width}: Generating {self.config.debate_config.num_agents} initial responses...")
            
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
            
            node = BeamNode(
                response=branch_response,
                score=0.0,  # Will be set by judge
                round=0,
                parent=None,
                trajectory=[branch_response],
                usage=[u for u in first_round_usage if u is not None]
            )
            self.node_to_branch[node_id] = branch_id  # Track which branch this node belongs to
            print(f"  ✓ Branch {branch_id+1} complete: {len(branch_response)} chars")
            
            return node
        
        # Run all branches in parallel
        branch_tasks = [create_branch(branch_id) for branch_id in range(self.config.beam_width)]
        nodes = await asyncio.gather(*branch_tasks)
        
        # Add all nodes to all_nodes
        for node in nodes:
            self.all_nodes.append(node)
        
        return nodes
    
    async def judge_nodes(self, question: str, nodes: List[BeamNode]) -> List[BeamNode]:
        """
        Judge all nodes using either scoring, ranking, or PRM based on config
        
        Args:
            question: Original question
            nodes: List of BeamNode to judge
        
        Returns:
            Same nodes with updated scores
        """
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
            
            print(f"\n⚖️  Judging {len(nodes)} responses (PRM)...")
            for i, node in enumerate(nodes, 1):
                print(f"  Response {i}: {node.score:.4f} (PRM)")
        
        elif self.config.judge_type == "ranking":
            print(f"\n⚖️  Ranking {len(nodes)} responses...")
            
            # Get all responses
            responses = [node.response for node in nodes]
            
            # Get ranking from judge
            ranking, explanation, usage = await judge_and_rank_responses(
                self.config.judge_model, question, responses
            )
            
            if usage:
                self.judge_usage.append(usage)
            
            # Assign scores based on ranking (best = highest score)
            num_candidates = len(nodes)
            for rank_position, node_idx in enumerate(ranking):
                score = float(num_candidates - rank_position)
                nodes[node_idx].score = score
                nodes[node_idx].judge_explanation = f"Rank: {rank_position + 1}/{num_candidates}\n{explanation}"
                print(f"    Node {node_idx + 1}: Rank {rank_position + 1}/{num_candidates} (Score: {score:.1f})")
        
        else:  # scoring
            print(f"\n⚖️  Judging {len(nodes)} responses...")
            
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
    
    def select_top_k(self, nodes: List[BeamNode], k: int) -> List[BeamNode]:
        """
        Select top k nodes based on judge scores
        
        Args:
            nodes: List of candidate nodes
            k: Number of nodes to keep
        
        Returns:
            Top k nodes
        """
        if self.config.select_best:
            # Deterministic: select top k by score
            sorted_nodes = sorted(nodes, key=lambda n: n.score, reverse=True)
            selected = sorted_nodes[:k]
        else:
            # Stochastic: sample k nodes weighted by scores
            # Normalize scores to probabilities
            total_score = sum(n.score for n in nodes)
            if total_score > 0:
                probs = [n.score / total_score for n in nodes]
                selected = random.choices(nodes, weights=probs, k=k)
            else:
                selected = random.sample(nodes, k=min(k, len(nodes)))
        
        judge_desc = "RANKING" if self.config.judge_type == "ranking" else ("PRM" if self.config.judge_type == "prm" else "SCORING")
        print(f"\n📊 Selected top {k} candidates ({judge_desc}):")
        for i, node in enumerate(selected, 1):
            if self.config.judge_type == "ranking":
                print(f"    {i}. Score: {node.score:.1f} (Round {node.round})")
            elif self.config.judge_type == "prm":
                print(f"    {i}. Score: {node.score:.4f} (Round {node.round}, PRM)")
            else:
                print(f"    {i}. Score: {node.score:.2f}/5 (Round {node.round})")
        
        return selected
    
    async def refine_branch(self, question: str, parent_node: BeamNode, parent_node_id: int, round_num: int) -> List[BeamNode]:
        """
        Refine a single branch by running beam_width refinement iterations in parallel
        Each iteration runs a full agent debate internally, creating beam_width child branches
        
        Args:
            question: Original question
            parent_node: Parent node representing this branch
            parent_node_id: ID of the parent node
            round_num: Current round number
        
        Returns:
            List of beam_width new BeamNode candidates from this branch
        """
        # Get the branch_id from parent
        parent_branch_id = self.node_to_branch[parent_node_id]
        
        # Get all previous agent responses for this branch
        parent_all_round_responses = self.branch_agent_responses[parent_branch_id]
        
        async def create_child(child_idx: int):
            """Create a single child branch by running refinement"""
            # Create a new branch_id for each child
            # print(f"    Creating child {child_idx+1}/{self.config.beam_width} for branch (round {round_num})...")
            
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
        child_tasks = [create_child(i) for i in range(self.config.beam_width)]
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
            
            child_node = BeamNode(
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
    
    async def run_refinement_round(self, question: str, parent_nodes: List[BeamNode], round_num: int) -> List[BeamNode]:
        """
        Run one refinement round: expand each parent branch into beam_width children
        All parent branches expand in parallel
        
        Args:
            question: Original question
            parent_nodes: Parent nodes (branches) to expand from
            round_num: Current round number
        
        Returns:
            List of all new BeamNode candidates (parent_count × beam_width)
        """
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}: Refinement ({len(parent_nodes)} branches × {self.config.beam_width} = {len(parent_nodes) * self.config.beam_width} children)")
        print(f"  Each child: {self.config.debate_config.num_agents} agents refine internally")
        print(f"{'='*60}")
        
        # For each parent branch, generate beam_width children in parallel
        async def expand_parent(parent, parent_idx):
            print(f"  Expanding branch {parent_idx+1}/{len(parent_nodes)} (score: {parent.score:.2f})...")
            # Find parent node ID by searching in all_nodes
            parent_node_id = self.all_nodes.index(parent)
            return await self.refine_branch(question, parent, parent_node_id, round_num)
        
        # Run all parent expansions in parallel
        expansion_tasks = [expand_parent(parent, i) for i, parent in enumerate(parent_nodes)]
        child_node_lists = await asyncio.gather(*expansion_tasks)
        
        # Flatten the list of lists
        all_new_nodes = []
        for child_nodes in child_node_lists:
            all_new_nodes.extend(child_nodes)
        
        print(f"  ✓ Generated {len(all_new_nodes)} candidate branches")
        
        return all_new_nodes
    
    async def search(self, question: str) -> Dict[str, Any]:
        """
        Run complete beam search process evaluation
        
        Args:
            question: Question to solve
        
        Returns:
            Dictionary with results including best trajectory
        """
        full_question = question
        
        if self.config.judge_type == "prm":
            judge_type_display = "PRM JUDGE"
        elif self.config.judge_type == "ranking":
            judge_type_display = "RANKING-BASED JUDGE"
        else:
            judge_type_display = "SCORING JUDGE"
        
        print(f"\n{'#'*60}")
        print(f"BEAM SEARCH PROCESS EVALUATION ({judge_type_display})")
        print(f"{'#'*60}")
        print(f"Question: {question[:100]}...")
        print(f"Beam Width: {self.config.beam_width}")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Model: {self.config.debate_config.model}")
        if self.config.judge_type == "prm":
            print(f"Judge: PRM ({self.config.prm_model_path.split('/')[-1]})")
        else:
            print(f"Judge: {self.config.judge_model} ({self.config.judge_type})")
        
        # Round 0: Initial generation
        current_nodes = await self.run_initial_round(full_question)
        current_nodes = await self.judge_nodes(full_question, current_nodes)
        
        # Keep top beam_width
        current_nodes = self.select_top_k(current_nodes, self.config.beam_width)
        
        # Refinement rounds
        for round_num in range(1, self.config.num_rounds + 1):
            # Expand each node
            candidate_nodes = await self.run_refinement_round(full_question, current_nodes, round_num)
            
            # Judge all candidates
            candidate_nodes = await self.judge_nodes(full_question, candidate_nodes)
            
            # Select top beam_width
            current_nodes = self.select_top_k(candidate_nodes, self.config.beam_width)
        
        # Collect all final responses from all beams
        all_final_responses = []
        all_final_nodes = []
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: Collecting all beam responses")
        print(f"{'='*60}")
        print(f"Total final beams: {len(current_nodes)}")
        
        for beam_idx, node in enumerate(current_nodes):
            # Get branch_id for this node
            node_id = self.all_nodes.index(node)
            branch_id = self.node_to_branch[node_id]
            
            # Get all agent responses from this branch
            branch_responses = self.branch_agent_responses[branch_id][-1]  # Last round responses
            
            # Add all agent responses from this beam
            for agent_idx, response in enumerate(branch_responses):
                all_final_responses.append(response)
                all_final_nodes.append(node)  # Keep reference to parent node for metadata
            
            print(f"\n  Beam {beam_idx + 1}:")
            print(f"    Node score: {node.score:.4f}" if self.config.judge_type == "prm" else f"    Node score: {node.score:.2f}")
            print(f"    Agents in this beam: {len(branch_responses)}")
            for agent_idx in range(len(branch_responses)):
                print(f"      Agent {agent_idx + 1}: {len(branch_responses[agent_idx])} chars")
        
        # Select best final node (for reporting/metadata)
        best_node = max(current_nodes, key=lambda n: n.score)
        
        print(f"\n{'='*60}")
        print(f"SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Total final responses: {len(all_final_responses)} ({len(current_nodes)} beams × {self.config.debate_config.num_agents} agents)")
        print(f"Best scoring beam:")
        if self.config.judge_type == "ranking":
            print(f"  Score: {best_node.score:.1f} (rank-based)")
        elif self.config.judge_type == "prm":
            print(f"  Score: {best_node.score:.4f} (PRM)")
        else:
            print(f"  Score: {best_node.score:.2f}/5")
        print(f"  Round: {best_node.round}")
        print(f"  Trajectory Length: {len(best_node.get_full_trajectory())}")
        print(f"Total Nodes Explored: {len(self.all_nodes)}")
        
        return {
            "question": question,
            "best_node": best_node,
            "best_score": best_node.score,
            "best_trajectory": best_node.get_full_trajectory(),
            "best_response": best_node.response,
            "all_final_responses": all_final_responses,  # All agent responses from all final beams
            "all_final_nodes": all_final_nodes,  # Corresponding node references
            "all_nodes": self.all_nodes,
            "final_candidates": current_nodes,  # The beam nodes themselves
            "config": {
                "beam_width": self.config.beam_width,
                "num_rounds": self.config.num_rounds,
                "debate_config": self.config.debate_config.__dict__,
                "judge_model": self.config.judge_model
            },
            "total_nodes_explored": len(self.all_nodes),
            "judge_usage": self.judge_usage
        }
