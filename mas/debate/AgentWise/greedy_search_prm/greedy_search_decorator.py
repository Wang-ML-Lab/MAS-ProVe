"""
Agent-Wise Greedy Search using Decorator-based Parallel Search
This is a simplified version that uses the llm_parallel_search_decorator for candidate generation
"""

import asyncio
import sys
import os
from typing import Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add mas-process-eval to path
mas_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, mas_eval_path)

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
    Agent-Wise Greedy Search using decorator-based parallel search.
    
    Workflow (preserves AgentWise logic):
    1. Round 0: For each agent, call direct(client=...) 
       → Decorator internally: generates N candidates, judges them, returns best
    2. Refinement: For each agent, call debate_refine(client=...)
       → Decorator internally: generates N refinements, judges them, returns best
    3. Final: Judge between agents to select overall best
    
    Key difference from manual version:
    - No manual candidate generation loops
    - No manual judging of candidates
    - Decorator handles parallel generation + judging per agent
    - We still maintain per-agent independence by calling decorated function once per agent
    """
    
    def __init__(self, config: GreedySearchConfig):
        self.config = config
        self.all_responses = []  # Track responses across rounds
        # Initialize judge server client
        self.judge_client = BaseClient(
            host=self.config.judge_server_host,
            port=self.config.judge_server_port
        )
    
    async def search(self, question: str) -> Dict[str, Any]:
        """
        Run agent-wise greedy search with decorator-based parallel generation.
        
        Args:
            question: The problem to solve
            
        Returns:
            Dictionary with search results
        """
        print(f"\n{'='*60}")
        print(f"AGENT-WISE GREEDY SEARCH (Decorator Mode)")
        print(f"{'='*60}")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Agents: {self.config.debate_config.num_agents}")
        print(f"Decorator handles parallel candidate generation & judging per agent")
        
        num_agents = self.config.debate_config.num_agents
        
        # Track best response for each agent across rounds
        agent_responses = {}  # {round: [agent_responses]}
        all_usage = []
        
        # ============== ROUND 0: Initial Generation ==============
        print(f"\n{'='*60}")
        print(f"ROUND 0: Initial Generation")
        print(f"{'='*60}")
        print(f"  Calling direct() for each of {num_agents} agents...")
        print(f"  (Each call: decorator generates N candidates → judges → returns best)")
        
        # Call direct() once per agent - decorator handles candidate generation internally
        initial_tasks = []
        for agent_id in range(num_agents):
            print(f"    Agent {agent_id + 1}: Calling decorated direct()...")
            initial_tasks.append(
                direct(
                    model=self.config.debate_config.model,
                    question=question,
                    dataset=self.config.debate_config.dataset,
                    use_tools=self.config.debate_config.use_tools,
                    temperature=self.config.debate_config.temperature,
                    max_tokens=self.config.debate_config.max_tokens,
                    # Pass client and task_type to enable decorator
                    client=self.judge_client,
                    task_type=self.config.task_type
                )
            )
        
        # Execute all agent generations in parallel
        initial_results = await asyncio.gather(*initial_tasks)
        agent_responses[0] = [result[0] for result in initial_results]  # Extract response text
        all_usage.extend([result[1] for result in initial_results if result[1] is not None])
        
        print(f"  ✓ Round 0 complete: Each agent has best response (selected by decorator)")
        
        # ============== REFINEMENT ROUNDS ==============
        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}: Refinement")
            print(f"{'='*60}")
            print(f"  Calling debate_refine() for each of {num_agents} agents...")
            print(f"  (Each call: decorator generates N refinements → judges → returns best)")
            
            # Get current best responses for all agents
            current_responses = agent_responses[round_num - 1]
            
            # Call debate_refine() once per agent - decorator handles refinement candidates internally
            refine_tasks = []
            for agent_id in range(num_agents):
                # Get other agents' responses
                other_agent_responses = current_responses[:agent_id] + current_responses[agent_id+1:]
                
                print(f"    Agent {agent_id + 1}: Calling decorated debate_refine()...")
                refine_tasks.append(
                    debate_refine(
                        model=self.config.debate_config.model,
                        question=question,
                        original_cot_response=current_responses[agent_id],
                        other_agents_responses=other_agent_responses,
                        dataset=self.config.debate_config.dataset,
                        use_tools=self.config.debate_config.use_tools,
                        temperature=self.config.debate_config.temperature,
                        max_tokens=self.config.debate_config.max_tokens,
                        # Pass client and task_type to enable decorator
                        client=self.judge_client,
                        task_type=self.config.task_type
                    )
                )
            
            # Execute refinements - decorator handles parallel search + judging for each agent
            refine_results = await asyncio.gather(*refine_tasks)
            agent_responses[round_num] = [result[0] for result in refine_results]
            all_usage.extend([result[1] for result in refine_results if result[1] is not None])
            
            print(f"  ✓ Round {round_num} complete: Each agent refined (best selected by decorator)")
        
        # ============== FINAL SELECTION ==============
        print(f"\n{'='*60}")
        print(f"FINAL: Selecting best agent")
        print(f"{'='*60}")
        
        final_responses = agent_responses[self.config.num_rounds]
        
        # Send final responses to judge for ranking between agents
        final_trajectories = [
            {"context": "", "current-step": resp}
            for resp in final_responses
        ]
        
        result = self.judge_client.send_request(
            task_type=self.config.task_type,
            judge_type="judge",
            question=question,
            partial_trajectories=final_trajectories
        )
        
        best_agent_id = result["rankings"][0]
        best_response = final_responses[best_agent_id]
        
        print(f"  Agent {best_agent_id + 1} selected as best overall")
        print(f"\n{'='*60}")
        print(f"SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Best Agent: {best_agent_id + 1}/{num_agents}")
        print(f"Total Rounds: {self.config.num_rounds + 1}")
        
        return {
            'question': question,
            'best_response': best_response,
            'best_agent_id': best_agent_id,
            'all_agent_responses': agent_responses,  # All responses by round
            'final_responses': final_responses,  # Final round responses for all agents
            'total_rounds': self.config.num_rounds + 1,
            'usage': all_usage,
            'config': {
                'num_rounds': self.config.num_rounds,
                'num_agents': num_agents,
                'debate_config': self.config.debate_config.__dict__
            }
        }
