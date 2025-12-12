"""
Greedy Search Process Reward Model (PRM) System
Supports both scoring and ranking-based judges
"""

from common import SearchNode as GreedyNode  # Alias for backward compatibility
from .greedy_search import GreedySearchPRM, GreedySearchConfig

# Backward compatibility aliases
GreedySearchRankingPRM = GreedySearchPRM
GreedySearchRankingConfig = GreedySearchConfig

__all__ = ['GreedyNode', 'GreedySearchPRM', 'GreedySearchConfig', 'GreedySearchRankingPRM', 'GreedySearchRankingConfig']
