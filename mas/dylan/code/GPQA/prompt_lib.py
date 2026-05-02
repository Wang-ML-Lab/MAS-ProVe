TEMPERATURE = 1.0
MAX_TOKENS = 2048

SYSTEM_PROMPT_GPQA = "It's a debate. Explain your reasons at each round thoroughly.\nAll questions are single choice."

ROLE_MAP = {
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
    "Researcher": "You are a meticulous scientific researcher. You are good at verifying claims, checking experimental logic, and identifying the most defensible answer.",
    "Physicist": "You are a physicist. You are good at physical reasoning, quantitative analysis, mechanics, electromagnetism, thermodynamics, and scientific modeling.",
    "Chemist": "You are a chemist. You are good at molecular reasoning, reactions, stoichiometry, spectroscopy, and laboratory interpretation.",
    "Biologist": "You are a biologist. You are good at genetics, evolution, cell biology, physiology, and experimental reasoning.",
    "Astronomer": "You are an astronomer. You are good at celestial mechanics, astrophysics, observational reasoning, and interpreting scientific evidence.",
    "Doctor": "You are a doctor. You are good at human biology, medicine, physiology, pathology, and evidence-based reasoning.",
    "Engineer": "You are an engineer. You are good at applying scientific principles, system analysis, estimation, and practical problem solving.",
    "Mathematician": "You are a mathematician. You are good at logical reasoning, symbolic manipulation, and careful problem solving.",
}


def construct_ranking_message(responses, question, qtype):
    prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

    for aid, aresponse in enumerate(responses, 1):
        response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
    return {"role": "user", "content": prefix_string}


def construct_message(responses, question, qtype):
    if len(responses) == 0:
        prefix_string = "Here is the question:\n" + question + "\n\nPut your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D)."
        return {"role": "user", "content": prefix_string}

    prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

    for aid, aresponse in enumerate(responses, 1):
        response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D). Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
    return {"role": "user", "content": prefix_string}