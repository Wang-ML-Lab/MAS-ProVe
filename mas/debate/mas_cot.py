from mas_proceval.mas.mas_base import MASBase
from mas_proceval.clients.client_base import BaseClient
from typing import Dict, Any, List, Tuple
from cot import CoTConfig

from cot import (
    extract_step_count,
    extract_step_content,
    check_stop_tag,
    parse_answer,
    cot_first_step as cot_first_step_naive,
    cot_next_step as cot_next_step_naive
)
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator

import asyncio
import cost_tracker


class MASCoT(MASBase):
    def __init__(self, example: Dict, example_id: int, cot_config: CoTConfig, benchmark: str, task_type: str = "math",
                 question_formatter=None, evaluate_fn=None):
        super().__init__()
        self.example = example
        self.example_id = example_id
        self.cot_config = cot_config
        self.benchmark = benchmark
        self.task_type = task_type
        self.question_formatter = question_formatter
        self.evaluate_fn = evaluate_fn

    async def run(self):
        print("Running MASCoT.run()")
        example, example_id, cot_config = self.example, self.example_id, self.cot_config

        print(f"Processing problem {example_id + 1} with CoT")
        
        # Use custom question formatter (required)
        if self.question_formatter:
            question = self.question_formatter(example)
        else:
            # Generic fallback - just use 'Question' or 'problem' field
            question = example.get('Question', example.get('problem', str(example)))
        
        # Run chain of thought
        cot_result, log = await self.chain_of_thought(
            model=cot_config.model,
            question=question,
            dataset=cot_config.dataset,
            use_tools=cot_config.use_tools,
            max_steps=cot_config.max_steps,
            temperature=cot_config.temperature
        )

        expected_answer = example.get('answer', example.get('Final answer', ''))
        
        # Use evaluation function (required from benchmark evaluator)
        if not self.evaluate_fn:
            raise ValueError("evaluate_fn must be provided by the benchmark evaluator")
        
        evaluation = self.evaluate_fn(cot_result["answer"], expected_answer)
        
        # Compute cost from usage data
        usage_list = cot_result.get("usage", [])
        total_cost = 0.0
        if usage_list:
            for usage in usage_list:
                if usage:
                    cost = cost_tracker.compute_cost(
                        cot_config.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    total_cost += cost
        
        return {
            "example_id": example_id,
            "problem_id": example_id,
            "problem": example.get('problem') or example.get('Question', str(example)),
            "cot_result": {
                "steps": cot_result["steps"],
                "step_count": len(cot_result["steps"]),
                "planned_steps": log[0].get("planned_steps", len(cot_result["steps"])),
                "full_response": cot_result["response"],
                "final_answer": cot_result["answer"],
            },
            "evaluation": evaluation,
            "expected_answer": expected_answer,
            "cost": total_cost,
            "usage": usage_list,
            "tool_calls": cot_result.get("tool_calls", [])
        }

    async def chain_of_thought(
        self,
        model: str,
        question: str,
        dataset: str = "aime24",
        use_tools: bool = False,
        max_steps: int = 4,
        log: dict = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict]:
        """
        Run Chain of Thought reasoning step by step with parallel search decorators.
        
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
        
        task_type = self.task_type
        
        all_usage = []  # Track all API call usage
        all_tool_calls = []  # Track all tool calls
        steps = []  # Store all reasoning steps
        
        # First step - get step count and initial reasoning
        print("Generating first step...")
        first_step, step_count, usage, tool_calls = await self.cot_first_step(
            model=model,
            question=question,
            dataset=dataset,
            use_tools=use_tools,
            task_type=task_type,
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
            
            next_step, usage, tool_calls = await self.cot_next_step(
                model=model,
                question=question,
                previous_steps=steps,
                dataset=dataset,
                use_tools=use_tools,
                task_type=task_type,
                trajectory=steps
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

    @llm_parallel_search_decorator
    async def cot_first_step(self, *args, **kwargs):
        return await cot_first_step_naive(*args, **kwargs)

    @llm_parallel_search_decorator
    async def cot_next_step(self, *args, **kwargs):
        return await cot_next_step_naive(*args, **kwargs)
