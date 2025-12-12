"""
Cost tracking utilities for LLM debate benchmarks
Tracks per-question token usage and computes costs based on model pricing
"""

from typing import Dict, Any, Optional


# Pricing per 1M tokens (input, output) in USD
model_price_map = {
    "gpt-4o_chatgpt": {"input": 2.50, "output": 10.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Compute total cost for a single API call.
    
    Args:
        model: Model name
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
    """
    if model not in model_price_map:
        # Unknown model, return 0 cost but don't fail
        return 0.0
    
    prices = model_price_map[model]
    input_cost = (prompt_tokens / 1000000) * prices["input"]
    output_cost = (completion_tokens / 1000000) * prices["output"]
    
    return input_cost + output_cost


def extract_usage_from_response(response: Any) -> Optional[Dict[str, int]]:
    """
    Extract token usage from OpenAI API response.
    
    Args:
        response: OpenAI API response object
        
    Returns:
        Dict with prompt_tokens and completion_tokens, or None if not available
    """
    try:
        if hasattr(response, 'usage'):
            usage = response.usage
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
    except (AttributeError, KeyError):
        pass
    
    return None


def aggregate_costs(cost_list: list[float]) -> float:
    """
    Aggregate multiple cost entries.
    
    Args:
        cost_list: List of costs from compute_cost()
        
    Returns:
        Total aggregated cost in USD
    """
    return sum(cost_list)
