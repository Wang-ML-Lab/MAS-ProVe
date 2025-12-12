"""
Agent-Wise Greedy Search Process Reward Model (PRM) System
Evaluates each agent independently across rounds
"""

from common import SearchNode as GreedyNode  # Alias for backward compatibility
from .greedy_search import GreedySearchPRM, GreedySearchConfig

# Backward compatibility aliases - ranking functionality is now integrated via judge_type parameter
GreedySearchRankingPRM = GreedySearchPRM
GreedySearchRankingConfig = GreedySearchConfig

__all__ = ['GreedyNode', 'GreedySearchPRM', 'GreedySearchConfig', 'GreedySearchRankingPRM', 'GreedySearchRankingConfig']
