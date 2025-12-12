"""
GAIA specific prompts for LLM debate system
"""

# GAIA direct solving prompt
GAIA_DIRECT_PROMPT_GPT5 = """You are a general AI assistant. 
If you are asked for a number, dont use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, dont use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element
to be put in the list is a number or a string.

QUESTION: {question}

Use the following web search results to help you answer the question:
{web_search_results}

You can freely reason in your response, please: 
1. include the thinking process in your response, in a detailed and extended manner, and after that, 
2. enclose the final answer within <answer></answer> tags (pure number without units and explanations)

"""

GAIA_DIRECT_PROMPT_GPT4 = """You are a general AI assistant. If you are asked for a number, don’t use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element
to be put in the list is a number or a string.

QUESTION: {question}

{web_search_results}

You can freely reason in your response, please: 
1. include the thinking process in your response, in a detailed and extended manner, and after that, 
2. enclose the final answer within <answer></answer> tags (pure number without units and explanations)

"""

# GAIA debate refinement prompt
GAIA_REFINE_PROMPT = """You are an AI assistant. Using the solutions from other agents as additional information, refine your original solution to the given problem, step-by-step.

**ORIGINAL TASK:** {question}
Here are some web search results that might help you:
{web_search_results}

**YOUR ORIGINAL APPROACH:** {original_cot_response}

**OTHER ASSISTANTS' APPROACHES:** {other_agents_responses}

You can freely reason in your response, but please enclose the final, logical answer within <answer></answer> tags.
"""

# Prompt selection function
def get_gaia_prompts(model: str):
    """Get appropriate GAIA prompts based on model type"""
    if "gpt-5" in model.lower():
        direct_prompt = GAIA_DIRECT_PROMPT_GPT5
    else:
        direct_prompt = GAIA_DIRECT_PROMPT_GPT4
    
    return {
        "direct": direct_prompt,
        "refine": GAIA_REFINE_PROMPT
    }