"""
Judge prompts for evaluating agent responses
"""


PROMPT_VERIFY_SYSTEM = """
Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user prompt displayed below. You will be given the assistant's response.

When evaluating the assistant's response, identify any mistakes or inaccurate information. Be as objective as possible. Avoid any biases, such as order of responses, length, or stylistic elements like formatting.

Before providing an your final verdict, think through the judging process and output your thoughts as an explanation

After providing your explanation, you must output a score as any floating point number between 0.0 to 5.0, where 0.0 represents you are completely certain that the response is incorrect and 5.0 represents you are completely certain that the response is correct. You can use decimal precision (e.g., 3.2, 4.7) to provide nuanced scores and avoid ties. Please enclose your score in <score> and </score> tags.
""".strip()


PROMPT_VERIFY = """
<|User Prompt|>
{question}
<|The Start of Assistant's Answer|>
{response}
<|The End of Assistant's Answer|>
""".strip()


def render_verify_prompt(response, question, prompt_strategy='vanilla'):
    """
    Render judge verification prompt
    
    Args:
        response: The assistant's response to evaluate
        question: The original question/prompt
        prompt_strategy: Strategy for prompting (default: 'vanilla')
    
    Returns:
        List of message dictionaries for the judge model
    """
    if prompt_strategy == 'vanilla':
        sys_prompt = PROMPT_VERIFY_SYSTEM
    
    prompt_template = PROMPT_VERIFY
    prompt_formatted = prompt_template.format(question=question, response=response)
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt_formatted}
    ]
    
    return messages


def judge(model: str, question: str, trajectory: str):
    """
    Create judge prompt for evaluating a response
    
    Args:
        model: Model name (not used in vanilla strategy, kept for compatibility)
        question: The original question
        trajectory: The assistant's response/trajectory to evaluate
    
    Returns:
        List of message dictionaries for the judge model
    """
    return render_verify_prompt(trajectory, question, prompt_strategy='vanilla')
