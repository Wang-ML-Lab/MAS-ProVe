"""
Beam Search Process Reward Model (PRM) Module

This module provides beam search-based process evaluation for LLM debate systems.
"""

from common import SearchNode as BeamNode  # Alias for backward compatibility
from .beam_search import BeamSearchPRM, BeamSearchConfig

# Backward compatibility: ranking versions are now the same as base versions with judge_type="ranking"
BeamSearchRankingPRM = BeamSearchPRM
BeamSearchRankingConfig = BeamSearchConfig

__all__ = ['BeamNode', 'BeamSearchPRM', 'BeamSearchConfig', 'BeamSearchRankingPRM', 'BeamSearchRankingConfig']
