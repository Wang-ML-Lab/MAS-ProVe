from mas_proceval.mas.mas_base import MASBase
from mas_proceval.clients.client_base import BaseClient
from typing import Dict, Any
from llm_debate_tool_call import DebateConfig

from llm_debate_tool_call import (
    parse_answer,
    debate as debate_naive, 
    direct as direct_naive, 
    debate_refine as debate_refine_naive
)
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator

import asyncio
import random
import re
import cost_tracker

class MASDebate(MASBase):
    def __init__(self, example: Dict, example_id: int, debate_config: DebateConfig, benchmark: str, task_type: str = "math", 
                 question_formatter=None, evaluate_fn=None):
        super().__init__()
        self.example = example
        self.example_id = example_id
        self.debate_config = debate_config
        self.benchmark = benchmark
        self.task_type = task_type
        self.question_formatter = question_formatter
        self.evaluate_fn = evaluate_fn

    async def run(self):
        print("Running MASDebate.run()")
        example, example_id, debate_config = self.example, self.example_id, self.debate_config

        print(f"Processing problem {example_id + 1}")
        # Use custom question formatter (required)
        if self.question_formatter:
            question = self.question_formatter(example)
        else:
            # Generic fallback - just use 'Question' or 'problem' field
            question = example.get('Question', example.get('problem', str(example)))
        
        debate_result, log = await self.debate(
            model=debate_config.model,
            question=question,
            num_agents=debate_config.num_agents,
            num_rounds=debate_config.num_rounds,
            dataset=self.benchmark,
            temperature=debate_config.temperature,
            max_tokens=debate_config.max_tokens
        )

        all_round_responses = log[0]["refine_results"]
        final_responses = all_round_responses[-1]
        expected_answer = example.get('answer')
        
        # Use evaluation function (required from benchmark evaluator)
        if not self.evaluate_fn:
            raise ValueError("evaluate_fn must be provided by the benchmark evaluator")
        
        evaluations = [{
            "agent_id": j,
            "answer": response,
            "evaluation": self.evaluate_fn(response, expected_answer)
        } for j, response in enumerate(final_responses)]
        
        # Compute cost from usage data
        usage_list = debate_result.get("usage", [])
        total_cost = 0.0
        if usage_list:
            for usage in usage_list:
                if usage:
                    cost = cost_tracker.compute_cost(
                        debate_config.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    total_cost += cost
        
        return {
            "example_id": example_id,
            "problem_id": example_id,
            "problem": example['problem'],
            "debate_result": {
                "round_history": all_round_responses,
                "final_responses": final_responses,
                "selected_response": debate_result["response"],
                "selected_answer": debate_result["answer"],
            },
            "agent_evaluations": evaluations,
            "expected_answer": expected_answer,
            "cost": total_cost,
        }

    async def debate(self, 
                     model: str,
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

        task_type = self.task_type
        
        all_usage = []  # Track all API call usage
        all_tool_calls = []  # Track all tool calls

        if cot_response is None:
            # All agents generate initial responses
            first_round_tasks = [
                self.direct(model=model, 
                            question=question, 
                            dataset=dataset, 
                            use_tools=use_tools, 
                            # make sure to pass the client to the decorated functions.
                            task_type=task_type)
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
                self.direct(model=model, 
                            question=question, 
                            dataset=dataset, 
                            use_tools=use_tools, 
                            # make sure to pass the client to the decorated functions.
                            task_type=task_type) 
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
                self.debate_refine(model=model,
                                   question=question,
                                   original_cot_response=current_cot,
                                   other_agents_responses=all_round_responses[-1][:i] + all_round_responses[-1][i+1:],
                                   dataset=dataset,
                                   use_tools=use_tools,
                                   task_type=task_type, 
                                   **kwargs) 
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

    @llm_parallel_search_decorator
    async def direct(self, *args, **kwargs):
        return await direct_naive(*args, **kwargs)

    @llm_parallel_search_decorator
    async def debate_refine(self, *args, **kwargs):
        return await debate_refine_naive(*args, **kwargs)