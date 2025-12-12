"""
HumanEval specific prompts for LLM debate system
"""

# HumanEval direct solving prompt
HUMANEVAL_DIRECT_PROMPT_GPT5 = """You are an expert Python programmer. Complete the given Python function:

QUESTION: {question}

You can freely reason about the solution, please:
1. Think through the problem requirements and edge cases
2. Implement a correct and efficient solution
3. Return ONLY the complete function implementation in a Python code block

Format your response as:
```python
[Your complete function implementation here]
```

Do not include test cases or example usage, only the function implementation.
"""

HUMANEVAL_DIRECT_PROMPT_GPT4 = """You are an expert Python programmer. Complete the given Python function:

QUESTION: {question}

Please think through the implementation carefully:
1. Understand the function signature and requirements
2. Consider edge cases and constraints
3. Write clean, correct Python code

Return your solution as:
```python
[Your complete function implementation here]
```

Provide only the function implementation, no test cases or explanations outside the code block.
"""

# HumanEval debate refinement prompt
HUMANEVAL_REFINE_PROMPT = """You are an expert Python programmer. Using the solutions from other agents as additional information, refine your original solution to complete the Python function.

**QUESTION:** {question}
**YOUR ORIGINAL SOLUTION:** {original_cot_response}
**OTHER AGENTS' SOLUTIONS:** {other_agents_responses}

Review the other solutions and improve your implementation if needed. Consider:
1. Correctness of the logic
2. Edge cases handling
3. Code efficiency and clarity

Return your refined solution as:
```python
[Your complete function implementation here]
```

Provide only the function implementation in the code block.
"""

# Prompt selection function
def get_humaneval_prompts(model: str):
    """Get appropriate HumanEval prompts based on model type"""
    if "gpt-5" in model.lower():
        direct_prompt = HUMANEVAL_DIRECT_PROMPT_GPT5
    else:
        direct_prompt = HUMANEVAL_DIRECT_PROMPT_GPT4
    
    return {
        "direct": direct_prompt,
        "refine": HUMANEVAL_REFINE_PROMPT
    }
