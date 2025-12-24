"""
GAIA-specific prompt library for DyLAN framework
"""

# System prompts for different roles
ROLE_MAP_GAIA = {
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
    "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
    "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
    "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
    "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
    "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations.",
    "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
    "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
    "Logician": "You are a logician. You are good at logic, reasoning, and critical thinking. You can analyze arguments and identify logical fallacies."
}

SYSTEM_PROMPT_GAIA = """It's a debate. Answer the question by using the web-search tool if needed and answer to the best of your knowledge. You can only make one search per question. You must provide a final answer with whatever search results you have, and whatever you can infer from your own knowledge.
IMPORTANT: Put your final answer inside <answer></answer> tags.
For example: <answer>42</answer> or <answer>Paris</answer>"""


def construct_message_gaia(formers, question):
    """Construct message for GAIA agents to see other answers"""
    if len(formers) == 0:
        return {
            "role": "user", 
            "content": f"Question: {question}\n\nPlease answer this question. Put your final answer in <answer></answer> tags."
        }

    prefix_string = f"Question: {question}\n\nHere are answers from other agents:\n"
    
    for idx, former in enumerate(formers, 1):
        prefix_string += f"\nAgent {idx}: {former}\n"
    
    prefix_string += f"\n\nUsing these answers as reference, provide your own answer. Think critically - the other answers might be incorrect. Put your final answer in <answer></answer> tags. Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {len(formers)} scores in the form like [[1, 5, 2, ...]]."
    
    return {"role": "user", "content": prefix_string}


def construct_ranking_message_gaia(formers, question):
    """Construct ranking message for GAIA"""
    if len(formers) == 0:
        return {"role": "user", "content": "Please verify your answer."}

    prefix_string = f"Question: {question}\n\nHere are different answers:\n"
    
    for idx, former in enumerate(formers, 1):
        prefix_string += f"\nAnswer {idx}: {former}\n"
    
    prefix_string += f"\nPlease rank the best {min(2, len(formers))} answers based on accuracy. Respond with [1,2] or similar format at the end."
    
    return {"role": "user", "content": prefix_string}
