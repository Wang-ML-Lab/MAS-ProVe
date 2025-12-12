"""
LLM Debate System for Multi-Agent Reasoning
Supports HumanEval, SWE-bench, GAIA, and AIME24 benchmarks
"""

import json
import os
import time
import asyncio
import random
import re
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI
from dataclasses import dataclass
import backoff
from prompts import prompt_manager
from ddgs import DDGS

async def web_search(search_query: str, max_results: int = 5):
    """Perform web search and return formatted results"""
    with DDGS() as ddgs:
        results = list(ddgs.text(search_query, max_results=max_results))
    
    formatted_output = "Search results:"
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Untitled')
        link = result.get('href', '')
        snippet = result.get('body', 'No content available.')
        formatted_output += f"--- SOURCE {i}: {title} ---\nURL: {link}\n\nCONTENT:\n{snippet}\n\n"
    
    return formatted_output


@dataclass
class DebateConfig:
    """Configuration for LLM debate system"""
    model: str = "gpt-5-mini"
    num_agents: int = 2
    num_rounds: int = 3
    temperature: float = 0.7
    max_tokens: int = 2048
    dataset: str = "aime24"  # Dataset-specific prompts: 'aime24', 'humaneval', 'swe', 'gaia'
    use_tools: bool = False  # Enable function calling (web search)


# Global async client
async_client = AsyncOpenAI()


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def call_llm(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> tuple:
    """Call LLM with proper parameters for different model types. Returns (content, usage)"""
    
    if model.startswith("gpt-5"):
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort = "minimal"
        )
    else:
        # Other models (GPT-4, etc.)
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    message = response.choices[0].message
    content = message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    } if hasattr(response, 'usage') and response.usage else None
    
    return content, usage


async def direct(model: str, question: str, dataset: str = "aime24", web_search_results: str = None, use_tools: bool = False, **kwargs) -> tuple:
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["direct"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Perform web search if tools enabled
    if use_tools:
        # print(f"  🔍 Agent performing web search...")
        web_search_results = await web_search(question)
        # print(f"  ✓ Search complete: {len(web_search_results)} characters")
    
    # Inject web search results if available
    if web_search_results and "{web_search_results}" in instruction:
        prompt = instruction.format(question=question, web_search_results=web_search_results)
        
    elif "{web_search_results}" in instruction:
        prompt = instruction.format(question=question, web_search_results="No web search performed.")
    else:
        prompt = instruction.format(question=question)
    # print(f"Direct Prompt:\n{prompt}\n")
    response, usage = await call_llm(model, prompt, **kwargs)
    # print(f"Direct Response:\n{response}\n")
    return response, usage


async def debate_refine(model: str, question: str, original_cot_response: str, other_agents_responses: list, dataset: str = "aime24", web_search_results: str = None, use_tools: bool = False, **kwargs) -> tuple:
    """Refine solution based on other agents' responses using dataset-specific prompts. Returns (response, usage)"""
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["refine"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get refine prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Perform web search if tools enabled
    if use_tools:
        # print(f"  🔍 Agent performing web search during refinement...")
        web_search_results = await web_search(question)
        # print(f"  ✓ Search complete: {len(web_search_results)} characters")
    
    # Format prompt with web search results if available
    if web_search_results and "{web_search_results}" in instruction:
        prompt = instruction.format(
            question=question,
            original_cot_response=original_cot_response,
            other_agents_responses="\n".join([f"Agent {i}: {response}" for i, response in enumerate(other_agents_responses)]),
            web_search_results=web_search_results
        )
        
    elif "{web_search_results}" in instruction:
        prompt = instruction.format(
            question=question,
            original_cot_response=original_cot_response,
            other_agents_responses="\n".join([f"Agent {i}: {response}" for i, response in enumerate(other_agents_responses)]),
            web_search_results="No web search performed."
        )
    else:
        prompt = instruction.format(
            question=question,
            original_cot_response=original_cot_response,
            other_agents_responses="\n".join([f"Agent {i}: {response}" for i, response in enumerate(other_agents_responses)])
        )
    # print(f"Refine Prompt:\n{prompt}\n")
    response, usage = await call_llm(model, prompt, **kwargs)
    # print(f"Refine Response:\n{response}\n")
    return response, usage


def parse_answer(response: str) -> str:
    """Extract answer from response text"""
    if "<answer>" in response and "</answer>" in response:
        start = response.find("<answer>") + len("<answer>")
        end = response.find("</answer>")
        return response[start:end].strip()
    else:
        # Fallback: take the last line or paragraph
        return response.split('\n')[-1].strip()


async def debate(model: str,
                 question: str,
                 cot_response: str = None,
                 num_agents: int = 2,
                 num_rounds: int = 3,
                 dataset: str = "aime24",
                 use_tools: bool = False,
                 log: dict = None,
                 **kwargs) -> tuple:
    """
    Run LLM debate with async execution
    
    Args:
        model: Model name to use
        question: Question to solve
        cot_response: Optional initial response from one agent
        num_agents: Number of agents in debate
        num_rounds: Number of refinement iterations
        dataset: Dataset name for prompts
        use_tools: Whether to enable web search (performed once before debate)
        log: Dictionary to store logs
        **kwargs: Additional arguments for LLM calls
    
    Returns:
        Tuple of (result_dict, log_dict)
    """
    log = log if log is not None else {}
    index = len(log)
    log[index] = {}
    
    all_usage = []  # Track all API call usage
    web_searches = []  # Track all web searches performed

    if cot_response is None:
        # All agents generate initial responses
        first_round_tasks = [
            direct(model=model, question=question, dataset=dataset, use_tools=use_tools, **kwargs) 
            for _ in range(num_agents)
        ]
        first_round_results = await asyncio.gather(*first_round_tasks)
        first_round_cot_responses = [result[0] for result in first_round_results]
        first_round_usage = [result[1] for result in first_round_results]
        all_usage.extend([u for u in first_round_usage if u is not None])
        all_round_responses = [first_round_cot_responses]
    else:
        # One agent starts with given response, others generate fresh responses
        first_round_tasks = [
            direct(model=model, question=question, dataset=dataset, use_tools=use_tools, **kwargs) 
            for _ in range(num_agents-1)
        ]
        first_round_results = await asyncio.gather(*first_round_tasks)
        first_round_cot_responses = [result[0] for result in first_round_results]
        first_round_usage = [result[1] for result in first_round_results]
        all_usage.extend([u for u in first_round_usage if u is not None])
        all_round_responses = [[cot_response] + first_round_cot_responses]

    # Refinement rounds
    for iteration in range(num_rounds):
        print(f"Refinement iteration {iteration + 1}/{num_rounds}")
        refine_tasks = [
            debate_refine(
                model=model,
                question=question,
                original_cot_response=current_cot,
                other_agents_responses=all_round_responses[-1][:i] + all_round_responses[-1][i+1:],
                dataset=dataset,
                use_tools=use_tools,
                **kwargs
            )
            for i, current_cot in enumerate(all_round_responses[-1])
        ]
        refine_results = await asyncio.gather(*refine_tasks)
        refine_responses = [result[0] for result in refine_results]
        refine_usage = [result[1] for result in refine_results]
        all_usage.extend([u for u in refine_usage if u is not None])
        all_round_responses.append(refine_responses)

    # Randomly pick one from the last round of responses
    random_response = random.choice(all_round_responses[-1])
    answer = parse_answer(random_response)

    log[index].update({
        "refine_results": all_round_responses,
        "num_agents": num_agents,
        "num_rounds": num_rounds,
        "final_answer": answer,
        "usage": all_usage,
        "web_searches": web_searches
    })

    return {"response": random_response, "answer": answer, "usage": all_usage, "web_searches": web_searches}, log

def save_results(results: List[Dict], output_file: str):
    """Save debate results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")