"""
LLM Debate System for Multi-Agent Reasoning
Supports GAIA, and AIME benchmarks
"""

import json
import os
import time
import asyncio
import random
import re
import sys
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI
from dataclasses import dataclass
import backoff
from prompts import prompt_manager
from ddgs import DDGS

# Add mas-process-eval to path for decorator import
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
class DebateConfig:
    """Configuration for LLM debate system"""
    model: str = "gpt-5-mini"
    num_agents: int = 3
    num_rounds: int = 2
    temperature: float = 0.7
    max_tokens: int = 2048
    dataset: str = "aime24"  # Dataset-specific prompts: 'aime24', 'humaneval', 'swe', 'gaia'
    use_tools: bool = False  # Enable function calling (web search) - set per benchmark


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
        # Add assistant's response to messages
        messages.append(response_message)
        
        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"  🔧 Tool call: {function_name}({function_args})")
            
            # Execute the function
            if function_name in TOOL_FUNCTIONS:
                function_response = TOOL_FUNCTIONS[function_name](**function_args)
                
                # Track tool call info
                tool_calls_info.append({
                    "function": function_name,
                    "arguments": function_args,
                    "response_length": len(str(function_response))
                })
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                })
            else:
                print(f"  ⚠️  Warning: Function {function_name} not found")
        
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
        
        # Add second call usage
        if hasattr(second_response, 'usage') and second_response.usage:
            usage["prompt_tokens"] += second_response.usage.prompt_tokens
            usage["completion_tokens"] += second_response.usage.completion_tokens
            usage["total_tokens"] += second_response.usage.total_tokens
        
        content = second_response.choices[0].message.content
    else:
        content = response_message.content
    
    return content, usage, tool_calls_info


async def direct(model: str, question: str, dataset: str = "aime24", use_tools: bool = False, **kwargs) -> tuple:
    """Generate initial solution for a problem using dataset-specific prompts. 
    
    If 'client' and 'task_type' are provided in kwargs, the decorator will:
    - Generate MAX_PARALLEL_SEARCH_CALLS candidates in parallel
    - Send them to judge server for ranking
    - Return only the best one
    
    Otherwise, generates a single response normally.
    
    Returns (response, usage, tool_calls_info)
    """
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["direct"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    prompt = instruction.format(question=question)
    response, usage, tool_calls_info = await call_llm(model, prompt, use_tools=use_tools, **kwargs)
    
    return response, usage, tool_calls_info


async def debate_refine(model: str, question: str, original_cot_response: str, other_agents_responses: list, dataset: str = "aime24", use_tools: bool = False, **kwargs) -> tuple:
    """Refine solution based on other agents' responses using dataset-specific prompts.
    
    If 'client' and 'task_type' are provided in kwargs, the decorator will:
    - Generate MAX_PARALLEL_SEARCH_CALLS refinement candidates in parallel
    - Send them to judge server for ranking
    - Return only the best one
    
    Otherwise, generates a single refinement normally.
    
    Returns (response, usage, tool_calls_info)
    """
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["refine"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get refine prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    prompt = instruction.format(
        question=question,
        original_cot_response=original_cot_response,
        other_agents_responses="\n".join([f"Agent {i}: {response}" for i, response in enumerate(other_agents_responses)])
    )
    
    response, usage, tool_calls_info = await call_llm(model, prompt, use_tools=use_tools, **kwargs)
    return response, usage, tool_calls_info


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
                 num_agents: int = 3,
                 num_rounds: int = 2,
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
    all_tool_calls = []  # Track all tool calls

    if cot_response is None:
        # All agents generate initial responses
        first_round_tasks = [
            direct(model=model, question=question, dataset=dataset, use_tools=use_tools, **kwargs) 
            for _ in range(num_agents)
        ]
        first_round_results = await asyncio.gather(*first_round_tasks)
        first_round_cot_responses = [result[0] for result in first_round_results]
        first_round_usage = [result[1] for result in first_round_results]
        first_round_tool_calls = [result[2] for result in first_round_results]
        
        all_usage.extend([u for u in first_round_usage if u is not None])
        all_tool_calls.extend([tc for tc in first_round_tool_calls if tc])
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
        first_round_tool_calls = [result[2] for result in first_round_results]
        
        all_usage.extend([u for u in first_round_usage if u is not None])
        all_tool_calls.extend([tc for tc in first_round_tool_calls if tc])
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
        refine_tool_calls = [result[2] for result in refine_results]
        
        all_usage.extend([u for u in refine_usage if u is not None])
        all_tool_calls.extend([tc for tc in refine_tool_calls if tc])
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
        "tool_calls": all_tool_calls  # Log all tool calls
    })

    return {"response": random_response, "answer": answer, "usage": all_usage, "tool_calls": all_tool_calls}, log


class LLMDebateSystem:
    """Main LLM Debate system using async architecture"""
    
    def __init__(self, config: DebateConfig):
        self.config = config
    
    async def debate_async(self, question: str, context: str = "") -> Dict[str, Any]:
        """Run full debate process on a single question (async)"""
        print(f"Starting async debate with {self.config.num_agents} agents for {self.config.num_rounds} iterations")
        
        # Format question with context if provided
        full_question = f"{question}\n\nContext: {context}" if context else question
        
        # Run the debate
        result, log = await debate(
            model=self.config.model,
            question=full_question,
            num_agents=self.config.num_agents,
            num_rounds=self.config.num_rounds,
            dataset=self.config.dataset,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Extract all responses from the debate log
        all_round_responses = log[0]["refine_results"]
        final_responses = all_round_responses[-1]
        
        return {
            "question": question,
            "context": context,
            "config": self.config.__dict__,
            "round_history": all_round_responses,
            "final_responses": final_responses,
            "selected_response": result["response"],
            "selected_answer": result["answer"],
            "log": log
        }
    
    def debate(self, question: str, context: str = "") -> Dict[str, Any]:
        """Synchronous wrapper for async debate"""
        return asyncio.run(self.debate_async(question, context))
    
    def extract_answers(self, responses: List[str]) -> List[str]:
        """Extract answers from response texts"""
        return [parse_answer(response) for response in responses]


def save_results(results: List[Dict], output_file: str):
    """Save debate results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")