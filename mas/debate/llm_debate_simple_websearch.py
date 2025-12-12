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
    
    formatted_output = "Search results:\n\n"
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Untitled')
        link = result.get('href', '')
        snippet = result.get('body', 'No content available.')
        formatted_output += f"--- SOURCE {i}: {title} ---\nURL: {link}\n\nCONTENT:\n{snippet}\n\n"
        formatted_output += "-" * 80 + "\n"
    
    return formatted_output


async def decide_and_search(question: str, context: str = None, cot_responses: list = None) -> tuple:
    """
    Use GPT-5-mini to decide if web search is needed and generate search query if yes.
    
    Args:
        question: The main question
        context: Optional context (for initial search decision)
        cot_responses: Optional list of existing responses (for refinement round decision)
    
    Returns:
        Tuple of (search_results_text or None, decision_info)
    """
    # Build decision prompt
    if cot_responses:
        # For refinement rounds - include existing responses
        decision_prompt = f"""You are a research assistant deciding whether additional web search is needed.

Question: {question}

Current responses from agents:
{chr(10).join([f"Agent {i+1}: {resp[:500]}..." for i, resp in enumerate(cot_responses)])}

Based on the question and the current responses, decide if a web search would provide helpful additional information.

Consider:
1. Are the current responses missing factual information that could be found online?
2. Does the question require current events, specific data, or verifiable facts?
3. Would web search significantly improve the answer quality?

If web search is needed:
- Output a focused search query between <query> and </query> tags
- Make the query specific and relevant to what's missing

If web search is NOT needed:
- Output <query>Not needed</query>

Your decision:"""
    else:
        # For initial round - just the question
        decision_prompt = f"""You are a research assistant deciding whether web search is needed to answer a question.

Question: {question}

Decide if this question requires web search to provide a good answer.

Consider:
1. Does it ask about current events, recent information, or specific facts?
2. Does it require data that changes over time (prices, statistics, etc.)?
3. Does it reference specific entities, places, or events that need verification?
4. Is it a general reasoning/math problem that doesn't need external facts?

If web search is needed:
- Output a focused search query between <query> and </query> tags
- Make the query specific and relevant

If web search is NOT needed:
- Output <query>Not needed</query>

Your decision:"""
    
    # Call GPT-5-mini to make decision
    decision_response, usage = await call_llm(
        model="gpt-5-mini",
        prompt=decision_prompt,
        temperature=0.3,  # Low temperature for consistent decisions
        max_tokens=200
    )
    
    # Extract query from response
    if "<query>" in decision_response and "</query>" in decision_response:
        start = decision_response.find("<query>") + len("<query>")
        end = decision_response.find("</query>")
        query = decision_response[start:end].strip()
    else:
        query = "Not needed"
    
    # Perform search if needed
    if query and query.lower() != "not needed":
        print(f"  🔍 Web search decision: YES - Query: '{query}'")
        search_results = await web_search(query, max_results=5)
        decision_info = {
            "decision": "search_performed",
            "query": query,
            "reasoning": decision_response,
            "usage": usage
        }
        return search_results, decision_info
    else:
        print(f"  ⏭️  Web search decision: NO - Not needed")
        decision_info = {
            "decision": "search_not_needed",
            "query": None,
            "reasoning": decision_response,
            "usage": usage
        }
        return None, decision_info


@dataclass
class DebateConfig:
    """Configuration for LLM debate system"""
    model: str = "gpt-5-mini"
    num_agents: int = 3
    num_rounds: int = 2
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
            reasoning_effort="minimal"
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


async def direct(model: str, question: str, dataset: str = "aime24", use_tools: bool = False, **kwargs) -> tuple:
    """Generate initial solution for a problem using dataset-specific prompts. Returns (response, usage, search_info)"""
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["direct"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    web_search_results = None
    search_info = None
    
    # Decide and perform web search if tools enabled
    if use_tools:
        web_search_results, search_info = await decide_and_search(question)
    
    # Inject web search results if available
    if web_search_results and "{web_search_results}" in instruction:
        # print(f" Web search results are:", web_search_results)
        prompt = instruction.format(question=question, web_search_results=web_search_results)
    elif "{web_search_results}" in instruction:
        prompt = instruction.format(question=question, web_search_results="No web search performed.")
    else:
        prompt = instruction.format(question=question)
    
    response, usage = await call_llm(model, prompt, **kwargs)
    return response, usage, search_info


async def debate_refine(model: str, question: str, original_cot_response: str, other_agents_responses: list, dataset: str = "aime24", use_tools: bool = False, **kwargs) -> tuple:
    """Refine solution based on other agents' responses using dataset-specific prompts. Returns (response, usage, search_info)"""
    try:
        prompts = prompt_manager.get_prompts(dataset, model)
        instruction = prompts["refine"]
    except (ValueError, KeyError) as e:
        error_msg = f"ERROR: Could not get refine prompts for dataset '{dataset}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    web_search_results = None
    search_info = None
    
    # Decide and perform web search if tools enabled
    # Pass current responses to help decide if additional search is needed
    if use_tools:
        all_current_responses = [original_cot_response] + other_agents_responses
        web_search_results, search_info = await decide_and_search(
            question=question,
            cot_responses=all_current_responses
        )
    # Format prompt with web search results if available
    if web_search_results and "{web_search_results}" in instruction:
        # print(f" Web search results are:", web_search_results)
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
    
    response, usage = await call_llm(model, prompt, **kwargs)
    return response, usage, search_info


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
    all_search_decisions = []  # Track all search decisions

    if cot_response is None:
        # All agents generate initial responses
        first_round_tasks = [
            direct(model=model, question=question, dataset=dataset, use_tools=use_tools, **kwargs) 
            for _ in range(num_agents)
        ]
        first_round_results = await asyncio.gather(*first_round_tasks)
        first_round_cot_responses = [result[0] for result in first_round_results]
        first_round_usage = [result[1] for result in first_round_results]
        first_round_search_info = [result[2] for result in first_round_results]
        
        all_usage.extend([u for u in first_round_usage if u is not None])
        all_search_decisions.extend([s for s in first_round_search_info if s is not None])
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
        first_round_search_info = [result[2] for result in first_round_results]
        
        all_usage.extend([u for u in first_round_usage if u is not None])
        all_search_decisions.extend([s for s in first_round_search_info if s is not None])
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
        refine_search_info = [result[2] for result in refine_results]
        
        all_usage.extend([u for u in refine_usage if u is not None])
        all_search_decisions.extend([s for s in refine_search_info if s is not None])
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
        "search_decisions": all_search_decisions  # Log all search decisions
    })

    return {"response": random_response, "answer": answer, "usage": all_usage, "search_decisions": all_search_decisions}, log


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