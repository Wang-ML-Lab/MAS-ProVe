"""
Prompt manager for dataset-specific prompts
"""

from .aime24_prompts import get_aime24_prompts
from .gaia_prompts import get_gaia_prompts


class PromptManager:
    """Manages dataset-specific prompts for LLM debate system"""
    
    def __init__(self):
        self.prompt_getters = {
            "aime24": get_aime24_prompts,
            "aime25": get_aime24_prompts,
            "gaia": get_gaia_prompts
        }
    
    def get_prompts(self, dataset: str, model: str):
        """
        Get prompts for specific dataset and model
        
        Args:
            dataset: Dataset name ('aime24', 'humaneval', 'swe', 'gaia')
            model: Model name (used to select appropriate prompt variant)
            
        Returns:
            Dictionary with 'direct' and 'refine' prompt templates
        """
        if dataset not in self.prompt_getters:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(self.prompt_getters.keys())}")
        
        return self.prompt_getters[dataset](model)
    
    def get_available_datasets(self):
        """Get list of available datasets"""
        return list(self.prompt_getters.keys())


# Global instance for easy import
prompt_manager = PromptManager()