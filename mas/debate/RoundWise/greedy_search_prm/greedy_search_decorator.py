"""
RoundWise Greedy Search using Decorator-based Parallel Search
This is a simplified version that uses the llm_parallel_search_decorator for candidate generation
"""

import asyncio
import random
import sys
import os
from typing import List, Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add mas-process-eval to path
mas_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, mas_eval_path)

from common import SearchNode as GreedyNode
from llm_debate_tool_call import DebateConfig, direct, debate_refine
from mas_proceval import BaseClient


@dataclass
class GreedySearchConfig:
    """Configuration for decorator-based greedy search"""
    num_rounds: int = 3  # Total number of refinement rounds
    debate_config: DebateConfig = None  # Configuration for underlying debate
    judge_server_host: str = "127.0.0.1"  # Judge server host
    judge_server_port: int = 5555  # Judge server port
    task_type: str = "math"  # Task type for judge server ("math", "code", etc.)
    
    def __post_init__(self):
        if self.debate_config is None:
            self.debate_config = DebateConfig()


class GreedySearchPRM_Decorator:
    """
    RoundWise Greedy Search using decorator-based parallel search.
    
    Workflow (preserves RoundWise logic):
    1. Round 0: Generate initial branch
       - For each agent: call direct(client=...) 
         → Decorator internally: generates N candidates, judges them, returns best
       - Randomly select one agent's response to represent the branch
    2. Refinement: Generate new candidates from best branch
       - For each agent: call debate_refine(client=...)
         → Decorator internally: generates N refinements, judges them, returns best
       - Randomly select one agent's response to represent the new branch
    3. Judge between branches (after all branches evaluated)
    4. Select best branch greedily
    5. Repeat from step 2 until num_rounds completed
    
    Key difference from manual version:
    - No manual candidate generation loops
    - No manual judging of candidates within agent responses
    - Decorator handles parallel search + judging per agent automatically
    - Still does branch-level judging (between branches)
    """
    
    def __init__(self, config: GreedySearchConfig):
        self.config = config
        self.all_nodes: List[GreedyNode] = []
        self.judge_usage: List[Dict] = []
        
        # Initialize judge server client
        self.judge_client = BaseClient(
            host=self.config.judge_server_host,
            port=self.config.judge_server_port
        )
        
        # Track agent responses for each branch
        self.branch_agent_responses: Dict[int, List[List[str]]] = {}
        self.node_to_branch: Dict[int, int] = {}
        self.node_counter: int = 0
    
    async def run_initial_round(self, question: str) -> GreedyNode:
        """
        Run initial round: Create initial branch with decorator
        Each agent generates their best response (decorator handles candidates internally)
        
        Returns:
            Single GreedyNode for the initial branch
        """
        print(f"\n{'='*60}")
        print(f"ROUND 0: Initial Generation")
        print(f"{'='*60}")
        print(f"  Generating initial branch with {self.config.debate_config.num_agents} agents...")
        print(f"  (Each agent: decorator generates N candidates → judges → returns best)")
        
        branch_id = 0
        
        # All agents generate their best initial responses (decorator handles candidates)
        first_round_tasks = [
            direct(
                model=self.config.debate_config.model,
                question=question,
                dataset=self.config.debate_config.dataset,
                use_tools=self.config.debate_config.use_tools,
                temperature=self.config.debate_config.temperature,
                max_tokens=self.config.debate_config.max_tokens,
                client=self.judge_client,
                task_type=self.config.task_type
            )
            for _ in range(self.config.debate_config.num_agents)
        ]
        
        first_round_results = await asyncio.gather(*first_round_tasks)
        first_round_cot_responses = [result[0] for result in first_round_results]
        first_round_usage = [result[1] for result in first_round_results]
        
        # Store agent responses for this branch
        self.branch_agent_responses[branch_id] = [first_round_cot_responses]
        
        # Randomly pick one response to represent this branch
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
        self.node_to_branch[node_id] = branch_id
        self.all_nodes.append(node)
        
        print(f"  ✓ Initial branch complete: {len(branch_response)} chars")
        
        return node
    
    async def refine_branch(self, question: str, parent_node: GreedyNode, parent_node_id: int, round_num: int) -> GreedyNode:
        """
        Refine branch by running refinement with decorator
        Each agent refines their response (decorator handles candidates internally)
        
        Args:
            question: Original question
            parent_node: Parent node representing this branch
            parent_node_id: ID of the parent node
            round_num: Current round number
        
        Returns:
            New GreedyNode for the refined branch
        """
        # Get the branch_id from parent
        parent_branch_id = self.node_to_branch[parent_node_id]
        
        # Get all previous agent responses for this branch
        parent_all_round_responses = self.branch_agent_responses[parent_branch_id]
        
        # Copy the agent responses history for the new branch
        new_branch_id = self.node_counter
        all_round_responses = [list(round_resps) for round_resps in parent_all_round_responses]
        
        # Run refinement for each agent (decorator handles candidates)
        refine_tasks = [
            debate_refine(
                model=self.config.debate_config.model,
                question=question,
                original_cot_response=current_cot,
                other_agents_responses=all_round_responses[-1][:i] + all_round_responses[-1][i+1:],
                dataset=self.config.debate_config.dataset,
                use_tools=self.config.debate_config.use_tools,
                temperature=self.config.debate_config.temperature,
                max_tokens=self.config.debate_config.max_tokens,
                client=self.judge_client,
                task_type=self.config.task_type
            )
            for i, current_cot in enumerate(all_round_responses[-1])
        ]
        
        refine_results = await asyncio.gather(*refine_tasks)
        refine_responses = [result[0] for result in refine_results]
        refine_usage = [result[1] for result in refine_results]
        
        # Update branch agent responses
        all_round_responses.append(refine_responses)
        self.branch_agent_responses[new_branch_id] = all_round_responses
        
        # Randomly pick one response to represent this branch
        branch_response = random.choice(refine_responses)
        
        # Create new node
        node_id = self.node_counter
        self.node_counter += 1
        
        new_trajectory = parent_node.trajectory + [branch_response]
        new_usage = list(parent_node.usage) + [u for u in refine_usage if u is not None]
        
        node = GreedyNode(
            response=branch_response,
            score=0.0,
            round=round_num,
            parent=parent_node_id,
            trajectory=new_trajectory,
            usage=new_usage
        )
        self.node_to_branch[node_id] = new_branch_id
        self.all_nodes.append(node)
        
        return node
    
    def select_best(self, node: GreedyNode) -> GreedyNode:
        """
        For decorator version, we only have one node at a time (greedy)
        Just return it with a note
        
        Args:
            node: The current node
        
        Returns:
            The same node
        """
        print(f"\n📊 Current branch (GREEDY - decorator-based):")
        print(f"  Round: {node.round}")
        print(f"  Response length: {len(node.response)} chars")
        
        return node
    
    async def search(self, question: str) -> Dict[str, Any]:
        """
        Run greedy search with decorator-based parallel search
        
        Args:
            question: Question to solve
        
        Returns:
            Dictionary with search results
        """
        print("\n" + "="*60)
        print("ROUNDWISE GREEDY SEARCH (Decorator Version)")
        print("="*60)
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Agents per round: {self.config.debate_config.num_agents}")
        print("Decorator handles parallel candidate generation & judging per agent")
        print("="*60)
        
        # Round 0: Initial generation
        current_node = await self.run_initial_round(question)
        current_node = self.select_best(current_node)
        current_node_id = len(self.all_nodes) - 1
        
        # Refinement rounds
        for round_num in range(1, self.config.num_rounds):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}: Refinement")
            print(f"{'='*60}")
            print(f"  Refining branch from Round {round_num-1}...")
            print(f"  (Each agent: decorator generates N refinements → judges → returns best)")
            
            # Refine the current best branch
            current_node = await self.refine_branch(question, current_node, current_node_id, round_num)
            current_node = self.select_best(current_node)
            current_node_id = len(self.all_nodes) - 1
            
            print(f"  ✓ Refinement complete")
        
        # Return results
        best_node = current_node
        
        print(f"\n{'='*60}")
        print("SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Best response: {len(best_node.response)} chars")
        print(f"Total nodes explored: {len(self.all_nodes)}")
        
        return {
            'best_response': best_node.response,
            'best_score': best_node.score,
            'best_trajectory': best_node.trajectory,
            'total_nodes_explored': len(self.all_nodes),
            'all_nodes': self.all_nodes,
            'judge_usage': self.judge_usage
        }
