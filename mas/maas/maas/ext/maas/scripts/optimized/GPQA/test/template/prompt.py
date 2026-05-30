
GENERATE_SOLUTION_PROMPT = """
You are an expert in answering graduate-level multiple-choice questions across diverse domains.

For the following question, analyze each option carefully and select the correct answer.

Follow this process:
1. UNDERSTAND: Read the question carefully and identify what is being asked.
2. ANALYZE: Consider each option (A, B, C, D) and evaluate its correctness based on your knowledge.
3. REASON: Explain your reasoning for choosing the correct answer.
4. ANSWER: State your final answer as a single letter (A, B, C, or D).

Question: {input}

Provide your reasoning and final answer in the format:
Reasoning: [Your detailed reasoning]
Answer: [A/B/C/D]
"""


SC_ENSEMBLE_PROMPT = """
You are an expert judge tasked with selecting the most consistent and correct answer from multiple solutions.

Question: {problem}

Multiple Solutions:
{solutions}

Instructions:
1. Review each solution (A, B, C, D, ...)
2. Compare their reasoning and final answers.
3. Identify which solution provides the most accurate and well-reasoned answer.
4. Select the letter of the best solution.

Return the selected solution letter.
"""

REFINE_ANSWER_PROMPT = """
Given the question, refine your previous answer by reconsidering each option carefully.

1. Review the question and all available options (A, B, C, D).
2. Eliminate any obviously incorrect options.
3. Compare the remaining options to identify the most accurate answer.
4. Provide clear reasoning for your choice.
5. State your final answer as a single letter (A, B, C, or D).

Provide your refined reasoning and final answer in the format:
Reasoning: [Your detailed reasoning]
Answer: [A/B/C/D]
"""

GENERATE_COT_PROMPT = """
You are an expert in answering graduate-level multiple-choice questions. Use chain-of-thought reasoning to arrive at the correct answer.

Question: {input}

Instructions:
1. Think step-by-step through the problem.
2. Consider what each option (A, B, C, D) represents.
3. Use your domain knowledge to evaluate each option.
4. Eliminate incorrect answers by identifying their logical flaws.
5. Arrive at the most correct answer.
6. Provide clear, detailed reasoning.

Format your response as:
Step-by-step reasoning: [Your detailed reasoning]
Final Answer: [A/B/C/D]
"""


SELFREFINE_PROMPT = """
Review the previous solution and refine it if necessary.

Question: {input}

Previous Answer: {solution}

Instructions:
1. Carefully re-examine the question.
2. Verify the reasoning in the previous answer.
3. Check if any option was misunderstood or misinterpreted.
4. Confirm that the chosen answer (A, B, C, or D) is the most correct.
5. If you believe the previous answer is incorrect, provide your corrected reasoning.

Provide your refined reasoning and final answer in the format:
Reasoning: [Your detailed reasoning]
Answer: [A/B/C/D]
"""
