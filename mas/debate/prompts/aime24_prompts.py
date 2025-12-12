"""
AIME24 specific prompts for LLM debate system
"""

# AIME24 direct solving prompt
AIME24_DIRECT_PROMPT_GPT5 = """You are a precise math problem solver. Solve the given math problem step by step:

QUESTION: {question}

You can freely reason in your response, please: 
1. include the thinking process in your response, in a detailed and extended manner, and after that, 
2. enclose the final answer within <answer></answer> tags (pure number without units and explanations)
"""

AIME24_DIRECT_PROMPT_GPT4 = """You are a precise math problem solver. Solve the given math problem step by step:

QUESTION: {question}

Please extend your chain of thought as much as possible; the longer the chain of thought, the better.

You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
"""

# AIME24 debate refinement prompt
AIME24_REFINE_PROMPT = """You are a precise math problem solver. Using the solutions from other agents as additional information, refine your original solution to the given math problem, step-by-step.

**QUESTION:** {question}
**ORIGINAL SOLUTION:** {original_cot_response}
**OTHER AGENTS' SOLUTIONS:** {other_agents_responses}

You can freely reason in your response, but please enclose the final, numerical answer within <answer></answer> tags (pure number only, without units or explanations).
"""

# Prompt selection function
def get_aime24_prompts(model: str):
    """Get appropriate AIME24 prompts based on model type"""
    if "gpt-5" in model.lower():
        direct_prompt = AIME24_DIRECT_PROMPT_GPT5
    else:
        direct_prompt = AIME24_DIRECT_PROMPT_GPT4
    
    return {
        "direct": direct_prompt,
        "refine": AIME24_REFINE_PROMPT
    }