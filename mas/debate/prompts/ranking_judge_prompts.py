"""
Ranking-based judge prompts for evaluating multiple agent responses
Instead of scoring individually, the judge ranks all candidates from best to worst
"""


PROMPT_RANKING_SYSTEM = """
Please act as an impartial judge and evaluate the quality of multiple responses provided by AI assistants to the user prompt displayed below. You will be given several candidate responses.

Your task is to rank these responses from best to worst based on their correctness, reasoning quality, and completeness. Consider:
- Correctness of the final answer
- Quality and clarity of reasoning
- Completeness of the solution
- Logical consistency

Be as objective as possible. Avoid any biases such as length or stylistic elements like formatting.

Before providing your ranking, think through the evaluation process and output your thoughts as an explanation.

After providing your explanation, you must output the ranking as a comma-separated list of candidate numbers (e.g., "2,1,3" means candidate 2 is best, candidate 1 is second, candidate 3 is worst). Please enclose your ranking in <ranking> and </ranking> tags.
""".strip()


PROMPT_RANKING = """
<|User Prompt|>
{question}

{candidates_text}
""".strip()


def render_ranking_prompt(responses: list, question: str):
    """
    Render judge ranking prompt for multiple candidates
    
    Args:
        responses: List of assistant responses to rank
        question: The original question/prompt
    
    Returns:
        List of message dictionaries for the judge model
    """
    # Format all candidates
    candidates_parts = []
    for i, response in enumerate(responses, 1):
        candidates_parts.append(f"<|Candidate {i} Response|>\n{response}\n<|End of Candidate {i}|>")
    
    candidates_text = "\n\n".join(candidates_parts)
    
    prompt_formatted = PROMPT_RANKING.format(
        question=question,
        candidates_text=candidates_text
    )
    
    messages = [
        {"role": "system", "content": PROMPT_RANKING_SYSTEM},
        {"role": "user", "content": prompt_formatted}
    ]
    
    return messages


def judge_ranking(model: str, question: str, responses: list):
    """
    Create judge prompt for ranking multiple responses
    
    Args:
        model: Model name (kept for compatibility)
        question: The original question
        responses: List of responses to rank
    
    Returns:
        List of message dictionaries for the judge model
    """
    return render_ranking_prompt(responses, question)
