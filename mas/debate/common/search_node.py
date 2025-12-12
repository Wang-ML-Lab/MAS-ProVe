"""
Generic Search Node: Represents a single candidate solution in search trees
Used by both RoundWise and AgentWise process evaluation methods
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class SearchNode:
    """
    Generic node for process evaluation search trees.
    
    Each node contains:
    - response: The agent's response at this round/step
    - score: Judge's evaluation score (0-5 for scoring, rank for ranking)
    - judge_explanation: Judge's reasoning
    - round: Which round/step this node is from (0, 1, 2, ...)
    - parent: Reference to parent node (None for root)
    - trajectory: Full conversation history up to this point
    - usage: API usage statistics
    """
    response: str
    score: float = 0.0
    judge_explanation: str = ""
    round: int = 0
    parent: Optional['SearchNode'] = None
    trajectory: List[str] = None
    usage: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = [self.response]
        if self.usage is None:
            self.usage = []
    
    def get_full_trajectory(self) -> List[str]:
        """Get complete trajectory from root to this node"""
        if self.parent is None:
            return self.trajectory.copy()
        else:
            parent_trajectory = self.parent.get_full_trajectory()
            parent_trajectory.append(self.response)
            return parent_trajectory
    
    def __repr__(self):
        return f"SearchNode(round={self.round}, score={self.score:.2f}, response_len={len(self.response)})"
