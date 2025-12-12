"""
Agent-Wise Beam Search Process Reward Model (PRM) System
Maintains beam width for each agent independently across rounds
"""

from common import SearchNode as BeamNode  # Alias for backward compatibility
from .beam_search import BeamSearchPRM, BeamSearchConfig

# Backward compatibility aliases - ranking functionality is now integrated via judge_type parameter
BeamSearchRankingPRM = BeamSearchPRM
BeamSearchRankingConfig = BeamSearchConfig

__all__ = ['BeamNode', 'BeamSearchPRM', 'BeamSearchConfig', 'BeamSearchRankingPRM', 'BeamSearchRankingConfig']
