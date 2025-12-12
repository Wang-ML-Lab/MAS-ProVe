"""
GAIA specific prompts for LLM debate system with tool calling support
These prompts do not include web_search_results placeholders as tools are called automatically
"""

# GAIA direct solving prompt for GPT-5 with tool calling
GAIA_DIRECT_PROMPT_GPT5 = """You are a general AI assistant with access to web search capabilities.
If you are asked for a number, don't use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element
to be put in the list is a number or a string.

QUESTION: {question}

If you need to look up information to answer this question, you can use the web_search function available to you.
Call it with specific search queries when you need factual information, current data, or verification.

You can freely reason in your response, please: 
1. include the thinking process in your response, in a detailed and extended manner, and after that, 
2. enclose the final answer within <answer></answer> tags (pure number without units and explanations)

"""

# GAIA direct solving prompt for GPT-4 with tool calling
GAIA_DIRECT_PROMPT_GPT4 = """You are a general AI assistant with access to web search capabilities.
If you are asked for a number, don't use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element
to be put in the list is a number or a string.

QUESTION: {question}

If you need to look up information to answer this question, you can use the web_search function available to you.
Call it with specific search queries when you need factual information, current data, or verification.

You can freely reason in your response, please: 
1. include the thinking process in your response, in a detailed and extended manner, and after that, 
2. enclose the final answer within <answer></answer> tags (pure number without units and explanations)

"""

# GAIA debate refinement prompt with tool calling
GAIA_REFINE_PROMPT = """You are an AI assistant with access to web search capabilities. Using the solutions from other agents as additional information, refine your original solution to the given problem, step-by-step.

**ORIGINAL TASK:** {question}

**YOUR ORIGINAL APPROACH:** {original_cot_response}

**OTHER ASSISTANTS' APPROACHES:** {other_agents_responses}

If you need additional information to improve your answer, you can use the web_search function.
You can freely reason in your response, but please enclose the final, logical answer within <answer></answer> tags.
"""

# Prompt selection function
def get_gaia_tool_prompts(model: str):
    """Get appropriate GAIA prompts with tool calling support based on model type"""
    if "gpt-5" in model.lower():
        direct_prompt = GAIA_DIRECT_PROMPT_GPT5
    else:
        direct_prompt = GAIA_DIRECT_PROMPT_GPT4
    
    return {
        "direct": direct_prompt,
        "refine": GAIA_REFINE_PROMPT
    }
