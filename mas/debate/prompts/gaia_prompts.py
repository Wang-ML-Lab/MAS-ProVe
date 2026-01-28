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
2. enclose the final answer within <answer></answer> tags.

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
2. enclose the final answer within <answer></answer> tags.

"""

# GAIA debate refinement prompt with tool calling
GAIA_REFINE_PROMPT = """You are an AI assistant with access to web search capabilities. Using the solutions from other agents as additional information, refine your original solution to the given problem, step-by-step.

**ORIGINAL TASK:** {question}

**YOUR ORIGINAL APPROACH:** {original_cot_response}

**OTHER ASSISTANTS' APPROACHES:** {other_agents_responses}

If you need additional information to improve your answer, you can use the web_search function.
You can freely reason in your response, but please enclose the final, logical answer within <answer></answer> tags.
"""

# GAIA Chain of Thought - First Step
GAIA_COT_FIRST_PROMPT = """You are a general AI assistant with access to web search capabilities using step-by-step reasoning.
If you are asked for a number, don't use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.

QUESTION: {question}

This is the FIRST step of your solution. Please:
1. Analyze the task and determine how many steps you'll need to solve it (maximum 4 steps)
2. Provide the first reasoning step

If you need to look up information, you can use the web_search function available to you.

Format your response as:
Total steps: [number between 1-4]
<step>
[Your first reasoning step here - show your initial analysis, key information gathering, or first action]
</step>

Do NOT provide the final answer yet. Focus only on the first step of reasoning.
"""

# GAIA Chain of Thought - Next Steps
GAIA_COT_NEXT_PROMPT = """You are a general AI assistant with access to web search capabilities using step-by-step reasoning.

QUESTION: {question}

PREVIOUS STEPS:
{previous_steps}

Based on the previous steps, provide the NEXT step in your reasoning process.

If you need additional information, you can use the web_search function.

Format your response as:
<step>
[Your next reasoning step here - continue from where you left off]
</step>

If this is the final step, include the answer within <answer></answer> tags:
<answer>[final answer]</answer>

If you're done before all planned steps, you can include a <stop> tag to indicate completion.
"""

# Prompt selection function
def get_gaia_prompts(model: str):
    """Get appropriate GAIA prompts with tool calling support based on model type"""
    if "gpt-5" in model.lower():
        direct_prompt = GAIA_DIRECT_PROMPT_GPT5
    else:
        direct_prompt = GAIA_DIRECT_PROMPT_GPT4
    
    return {
        "direct": direct_prompt,
        "refine": GAIA_REFINE_PROMPT,
        "cot_first": GAIA_COT_FIRST_PROMPT,
        "cot_next": GAIA_COT_NEXT_PROMPT
    }

