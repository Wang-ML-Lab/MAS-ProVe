SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""


SELFREFINE_PROMPT = """
You are an assistant specialized in refining solutions to problems.

Problem:
{problem}

Solution:
{solution}

Instruction:
Analyze the above solution for any errors or suboptimal aspects. 
1. Verification: Does the reasoning and logic hold up under scrutiny? Are there any factual errors or logical gaps?
2. Completeness: Does the solution fully address all aspects of the question? Is any important information missing?
3. Format Check: Ensure the final answer follows these rules:
   - If it's a number: no commas, no units (like $ or %) unless specified
   - If it's a string: no articles, no abbreviations (e.g. for cities), digits in plain text unless specified
   - If it's a comma separated list: apply above rules based on element type
4. Clarity: Is the explanation clear and well-structured?
5. Refinement: Make iterative improvements to enhance correctness, completeness, and clarity. Provide the refined solution below.
"""

GENERATE_COT_PROMPT = '''
Reasoning Instruction
{instruction}

Current Problem: {input}

IMPORTANT - Answer Formatting Rules:
- If you are asked for a number, don't use commas to write your number and don't use units such as $ or percent sign unless specified otherwise.
- If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
- If you are asked for a comma separated list, apply the above rules depending on whether the element to be put in the list is a number or a string.

Solution Protocol:
1. Carefully read and understand the question. Identify what is being asked.
2. Break down the problem into smaller, manageable steps.
3. For each step, provide clear reasoning and explain your thought process.
4. Use relevant facts, data, or context to support your reasoning.
5. Verify your intermediate results and check for logical consistency.
6. Synthesize your findings into a clear, concise final answer following the formatting rules above.
7. Present your final answer enclosed in \boxed{} notation (e.g., \boxed{42} or \boxed{Paris}).
8. Double-check that your answer directly addresses the question asked and follows the proper format.

Step-by-Step Analysis:
'''