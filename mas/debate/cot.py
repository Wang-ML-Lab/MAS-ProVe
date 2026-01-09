"""
Chain of Thought (CoT) System for Step-by-Step Reasoning
Supports AIME and GAIA benchmarks with iterative step generation
"""

import json
import os
import sys
import asyncio
import random
import re
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import AsyncOpenAI
from dataclasses import dataclass
import backoff
from prompts import prompt_manager
from ddgs import DDGS

# Add mas-process-eval to path
mas_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, mas_eval_path)


def web_search(search_query: str, max_results: int = 5):
    """Perform web search and return formatted results (synchronous for tool calling)"""
    with DDGS() as ddgs:
        results = list(ddgs.text(search_query, max_results=max_results))
    
    formatted_output = "Search results:\n\n"
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Untitled')
        link = result.get('href', '')
        snippet = result.get('body', 'No content available.')
        formatted_output += f"--- SOURCE {i}: {title} ---\nURL: {link}\n\nCONTENT:\n{snippet}\n\n"
        formatted_output += "-" * 80 + "\n"
    
    return formatted_output


# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a web search using DuckDuckGo to find information on the internet. Use this when you need to look up current information, facts, or details that you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search query to look up on the web",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default: 5)",
                        "default": 5
                    },
                },
                "required": ["search_query"],
            },
        }
    }
]

TOOL_FUNCTIONS = {
    "web_search": web_search
}


@dataclass
class CoTConfig:
    """Configuration for Chain of Thought system"""
    model: str = "gpt-5-mini"
    temperature: float = 0.7
    max_tokens: int = 2048
    max_steps: int = 4  # Maximum number of reasoning steps
    dataset: str = "aime24"  # Dataset-specific prompts: 'aime24', 'gaia'
    use_tools: bool = False  # Enable function calling (web search)


# Global async client
async_client = AsyncOpenAI()


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def call_llm(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2048, use_tools: bool = False) -> tuple:
    """Call LLM with proper parameters for different model types. Returns (content, usage, tool_calls_info)"""
    
    messages = [{"role": "user", "content": prompt}]
    request_params = {
        "model": model,
        "messages": messages,
    }
    
    # Add tools if enabled
    if use_tools:
        request_params["tools"] = TOOLS
        request_params["tool_choice"] = "auto"
    
    # Add model-specific parameters
    if model.startswith("gpt-5"):
        request_params["reasoning_effort"] = "minimal"
    else:
        request_params["temperature"] = temperature
        request_params["max_tokens"] = max_tokens
    
    # Make initial API call
    response = await async_client.chat.completions.create(**request_params)
    
    # Track initial token usage
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    } if hasattr(response, 'usage') and response.usage else None
    
    # Handle tool calls if present
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    tool_calls_info = []
    
    if tool_calls:
        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            tool_calls_info.append({
                "function": function_name,
                "arguments": function_args
            })
            print(f"  🔧 Tool call: {function_name}({function_args})")
            if function_name in TOOL_FUNCTIONS:
                function_response = TOOL_FUNCTIONS[function_name](**function_args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
        
        # Make second API call with tool results
        second_request_params = {
            "model": model,
            "messages": messages,
        }
        
        if model.startswith("gpt-5"):
            second_request_params["reasoning_effort"] = "minimal"
        else:
            second_request_params["temperature"] = temperature
            second_request_params["max_tokens"] = max_tokens
        
        second_response = await async_client.chat.completions.create(**second_request_params)
        
        # Update usage with second call
        if hasattr(second_response, 'usage') and second_response.usage:
            if usage:
                usage["prompt_tokens"] += second_response.usage.prompt_tokens
                usage["completion_tokens"] += second_response.usage.completion_tokens
                usage["total_tokens"] += second_response.usage.total_tokens
            else:
                usage = {
                    "prompt_tokens": second_response.usage.prompt_tokens,
                    "completion_tokens": second_response.usage.completion_tokens,
                    "total_tokens": second_response.usage.total_tokens,
                }
        
        content = second_response.choices[0].message.content
    else:
        content = response_message.content
    
    return content, usage, tool_calls_info


def extract_step_count(response: str) -> int:
    """Extract the number of steps from first CoT response."""
    # Look for patterns like "steps: 3", "3 steps", "total steps: 4"
    patterns = [
        r'total\s+steps?\s*:?\s*(\d+)',
        r'steps?\s*:?\s*(\d+)',
        r'(\d+)\s+steps?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            return min(int(match.group(1)), 4)  # Cap at 4 steps
    
    # Default to 4 if not found
    return 4


def extract_step_content(response: str) -> str:
    """Extract step content from <step></step> tags."""
    if "<step>" in response and "</step>" in response:
        start = response.find("<step>") + len("<step>")
        end = response.find("</step>")
        return response[start:end].strip()
    else:
        # Fallback: return full response if tags not found
        return response.strip()


def check_stop_tag(response: str) -> bool:
    """Check if response contains <stop> tag indicating early completion."""
    return "<stop>" in response.lower()


def parse_answer(response: str) -> str:
    """Extract answer from response text"""
    if "<answer>" in response and "</answer>" in response:
        start = response.find("<answer>") + len("<answer>")
        end = response.find("</answer>")
        return response[start:end].strip()
    else:
        # Fallback: take the last line or paragraph
        return response.split('\n')[-1].strip()


async def cot_first_step(model: str, question: str, dataset: str = "aime24", use_tools: bool = False, **kwargs) -> Tuple[str, int, dict, list]:
    """
    Generate first step and get step count.
    
    Returns:
        Tuple of (step_content, step_count, usage, tool_calls_info)
    """
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["cot_first"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get CoT first step prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    prompt = instruction.format(question=question)
    response, usage, tool_calls_info = await call_llm(model, prompt, use_tools=use_tools, **kwargs)
    
    step_count = extract_step_count(response)
    step_content = extract_step_content(response)
    
    return step_content, step_count, usage, tool_calls_info


async def cot_next_step(model: str, question: str, previous_steps: List[str], dataset: str = "aime24", use_tools: bool = False, **kwargs) -> Tuple[str, dict, list]:
    """
    Generate next step based on previous steps.
    
    Returns:
        Tuple of (step_content, usage, tool_calls_info)
    """
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["cot_next"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get CoT next step prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Format previous steps
    previous_steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(previous_steps)])
    
    prompt = instruction.format(
        question=question,
        previous_steps=previous_steps_text
    )
    
    response, usage, tool_calls_info = await call_llm(model, prompt, use_tools=use_tools, **kwargs)
    step_content = extract_step_content(response)
    
    return step_content, usage, tool_calls_info


async def chain_of_thought(
    model: str,
    question: str,
    dataset: str = "aime24",
    use_tools: bool = False,
    max_steps: int = 4,
    log: dict = None,
    **kwargs
) -> Tuple[Dict[str, Any], Dict]:
    """
    Run Chain of Thought reasoning step by step.
    
    Args:
        model: Model name to use
        question: Question to solve
        dataset: Dataset name for prompts
        use_tools: Whether to enable web search
        max_steps: Maximum number of steps (capped at 4)
        log: Dictionary to store logs
        **kwargs: Additional arguments for LLM calls
    
    Returns:
        Tuple of (result_dict, log_dict)
    """
    log = log if log is not None else {}
    index = len(log)
    log[index] = {}
    
    all_usage = []  # Track all API call usage
    all_tool_calls = []  # Track all tool calls
    steps = []  # Store all reasoning steps
    
    # First step - get step count and initial reasoning
    print("Generating first step...")
    first_step, step_count, usage, tool_calls = await cot_first_step(
        model=model,
        question=question,
        dataset=dataset,
        use_tools=use_tools,
        **kwargs
    )
    
    steps.append(first_step)
    all_usage.append(usage)
    if tool_calls:
        all_tool_calls.extend(tool_calls)
    
    print(f"Step count: {step_count}, First step: {first_step[:100]}...")
    
    # Check for early stop
    if check_stop_tag(first_step):
        print("Stop tag detected in first step")
        step_count = 1
    
    # Generate remaining steps
    for step_num in range(1, min(step_count, max_steps)):
        print(f"Generating step {step_num + 1}/{step_count}...")
        
        next_step, usage, tool_calls = await cot_next_step(
            model=model,
            question=question,
            previous_steps=steps,
            dataset=dataset,
            use_tools=use_tools,
            **kwargs
        )
        
        steps.append(next_step)
        all_usage.append(usage)
        if tool_calls:
            all_tool_calls.extend(tool_calls)
        
        print(f"Step {step_num + 1}: {next_step[:100]}...")
        
        # Check for stop tag
        if check_stop_tag(next_step):
            print(f"Stop tag detected at step {step_num + 1}")
            break
    
    # Combine all steps into final response
    full_response = "\n\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
    
    # Extract final answer from last step
    answer = parse_answer(steps[-1])
    
    log[index].update({
        "steps": steps,
        "step_count": len(steps),
        "planned_steps": step_count,
        "final_answer": answer,
        "usage": all_usage,
        "tool_calls": all_tool_calls
    })
    
    return {
        "response": full_response,
        "answer": answer,
        "steps": steps,
        "usage": all_usage,
        "tool_calls": all_tool_calls
    }, log


def save_results(results: List[Dict], output_file: str):
    """Save CoT results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
