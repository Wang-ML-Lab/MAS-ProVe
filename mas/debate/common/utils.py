"""
Utility functions shared across all search strategies (RoundWise and AgentWise)
"""

import openai
from openai import AsyncOpenAI
import backoff
from typing import List, Dict, Tuple
import requests


# Global async client
async_client = AsyncOpenAI()

# Global summarization usage tracker
summarization_usage = []


def request_for_process_rewards(query: str, step_responses: List[str], model: str, api_url: str = "http://localhost:8000/pooling") -> List[float]:
    """
    Request process rewards from PRM API.
    
    Args:
        query: The question/prompt
        step_responses: List of response strings to score
        model: Path to the PRM model
        api_url: URL for the PRM API endpoint
        
    Returns:
        List of reward scores (one per response)
    """
    try:
        conversation_str = assemble_conversation_str(query, step_responses, model)
        prompt = {"model": model, "input": conversation_str}
        
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers=headers, json=prompt, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(
                f"Request failed with status code {response.status_code}: {response.text}")
        
        rewards = [x[1] for x in response.json()["data"][0]["data"]]
        
        assert len(rewards) == len(step_responses), "Rewards length mismatch with step responses"
        
        return rewards
    except Exception as e:
        print(f"Error requesting PRM rewards: {e}")
        return [0.0] * len(step_responses)


def assemble_conversation_str(query: str, step_responses: List[str], model: str) -> str:
    """
    Assemble conversation string for PRM model input.
    
    Args:
        query: The question/prompt
        step_responses: List of response strings
        model: Path to the PRM model
        
    Returns:
        Formatted conversation string
    """
    # should work for both Qwen2.5-Math-PRM-7B and Qwen2.5-Math-PRM-32B
    if "qwen2.5-math-prm" in model.lower():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        system = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(step_responses) + "<extra_0>"},
        ]
        
        conversation_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    return conversation_str


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def summarize_response(original_response: str, model: str = "gpt-5-mini") -> Tuple[str, Dict]:
    """
    Summarize a long response using a meta model for PRM evaluation.
    
    Args:
        original_response: The original response to summarize
        model: Model to use for summarization (default: gpt-5-nano)
        
    Returns:
        Tuple of (summarized_response, usage_dict)
    """
    summarization_prompt = f"""Your role is to act as a sophisticated summarization engine. Analyze the provided thoughts and answers, consolidate them into a comprehensive, and logically flowing chain-of-thought. Ensure the summary captures the complete problem understanding, all essential reasoning steps, and and put the final answer within <answer></answer> tags (pure number without units and explanations)

**ORIGINAL RESPONSE:** {original_response}"""
    
    messages = [
        {"role": "user", "content": summarization_prompt}
    ]
    
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort="minimal",
            verbosity = "low"
        )
        
        summarized_content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "model": model
        }
        
        # Track usage globally
        summarization_usage.append(usage)
        
        return summarized_content, usage
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        # Return original response if summarization fails
        return original_response, {"prompt_tokens": 0, "completion_tokens": 0, "model": model, "error": str(e)}


async def calculate_prm_score(
    prompt: str, 
    final_response: str,
    prm_model_path: str = None,
    api_url: str = "http://localhost:8000/pooling",
    enable_summarization: bool = False
) -> Tuple[float, Dict]:
    """
    Calculate Process Reward Model (PRM) score for a given solution.
    
    Args:
        prompt: The question/problem statement
        final_response: The response to score
        prm_model_path: Path to the PRM model (uses default if None)
        api_url: URL for the PRM API endpoint
        enable_summarization: Whether to use meta model for summarization (default: False)
    
    Returns:
        Tuple of (prm_score, usage_dict): 
            - prm_score: float between 0 and 1, or 0.0 if error occurs
            - usage_dict: summarization usage info (empty dict if no summarization)
    """
    if not prompt:
        print("Warning: No prompt provided")
        return 0.0, {}
    
    if not final_response:
        print("Warning: No response provided, using empty answer")
        return 0.0, {}
    
    answer = final_response.strip()
    original_length = len(answer)
    # print(f"Answer length for PRM: {original_length} chars")
    
    # Apply summarization if enabled
    summarization_usage_info = {}
    if enable_summarization:
        print(f"Summarizing response (original: {original_length} chars)...")
        summarized_answer, usage = await summarize_response(answer, model="gpt-5-mini")
        summarized_length = len(summarized_answer)
        # print(f"Summarized answer: {summarized_answer}")
        # print(f"Summarized to {summarized_length} chars (saved {original_length - summarized_length} chars)")
        # print(f"Summarization cost: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion tokens")
        summarization_usage_info = usage
    else:
        summarized_answer = answer
    
    try:
        # Use default model path if not provided
        if prm_model_path is None:
            prm_model_path = "/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B"
        
        rewards = request_for_process_rewards(
            query=prompt,
            step_responses=[summarized_answer],
            model=prm_model_path,
            api_url=api_url
        )
        prm_score = rewards[0] if rewards else 0.0
    except Exception as e:
        print(f"Error calculating PRM score: {e}")
        prm_score = 0.0
    
    return prm_score, summarization_usage_info


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def call_judge_llm(model: str, messages: List[Dict[str, str]]) -> tuple:
    """
    Call LLM judge with proper parameters. Returns (content, usage)
    
    Args:
        model: Model name to use for judging
        messages: List of message dictionaries for the chat API
        
    Returns:
        tuple: (response_content, usage_dict)
    """
    
    if model.startswith("gpt-5"):
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort="minimal"
        )
    else:
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages,
        )
    
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return content, usage


def parse_judge_score(response: str) -> float:
    """
    Extract judge score from response text.
    
    Args:
        response: Judge's response containing <score>X</score> tags
        
    Returns:
        float: Extracted score, or 0.0 if parsing fails
    """
    if "<score>" in response and "</score>" in response:
        start = response.find("<score>") + len("<score>")
        end = response.find("</score>")
        score_text = response[start:end].strip()
        try:
            return float(score_text)
        except ValueError:
            return 0.0
    return 0.0


def parse_ranking(response: str, num_candidates: int) -> List[int]:
    """
    Extract ranking from response text.
    
    Args:
        response: Judge's response containing ranking information
        num_candidates: Number of candidates being ranked
        
    Returns:
        List of candidate indices in order from best to worst (0-indexed)
        e.g., [1, 0, 2] means candidate 2 is best, candidate 1 is second, candidate 3 is worst
    """
    if "<ranking>" in response and "</ranking>" in response:
        start = response.find("<ranking>") + len("<ranking>")
        end = response.find("</ranking>")
        ranking_text = response[start:end].strip()
        
        try:
            # Parse comma-separated ranking (1-indexed from judge)
            ranks = [int(x.strip()) - 1 for x in ranking_text.split(",")]
            
            # Validate ranking
            if len(ranks) == num_candidates and set(ranks) == set(range(num_candidates)):
                return ranks
        except (ValueError, IndexError):
            pass
    
    # Fallback: return random ranking
    print(f"Warning: Could not parse ranking from judge response, using random ranking")
    import random
    fallback = list(range(num_candidates))
    random.shuffle(fallback)
    return fallback


def get_summarization_usage() -> List[Dict]:
    """
    Get all summarization usage records.
    
    Returns:
        List of usage dictionaries with token counts and model info
    """
    return summarization_usage


def reset_summarization_usage():
    """Reset the global summarization usage tracker."""
    global summarization_usage
    summarization_usage = []


def compute_summarization_cost(usage_list: List[Dict] = None) -> Dict[str, float]:
    """
    Compute total cost of summarization calls.
    
    Args:
        usage_list: List of usage dicts (uses global tracker if None)
        
    Returns:
        Dictionary with token counts and estimated cost
    """
    if usage_list is None:
        usage_list = summarization_usage
    
    total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage_list)
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in usage_list)
    
    # gpt-5-nano pricing (example rates - adjust as needed)
    # Assuming $0.15 per 1M prompt tokens, $0.60 per 1M completion tokens
    prompt_cost = total_prompt_tokens * 0.15 / 1_000_000
    completion_cost = total_completion_tokens * 0.60 / 1_000_000
    total_cost = prompt_cost + completion_cost
    
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": total_cost,
        "num_calls": len(usage_list)
    }
