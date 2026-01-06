
GENERATE_SOLUTION_PROMPT = """
You are an intelligent agent capable of solving complex tasks that require external information.
You have access to a 'web_search' tool. DO NOT hallucinate facts.

Follow this process:
1. ANALYSIS: Break down the user's request. Identify what facts, numbers, or dates you need to find.
2. SEARCH: Use the search tool to find this specific information. 
   - If the first search is insufficient, perform a new search with different keywords.
   - You can search multiple times.
3. VERIFICATION: Compare data from search results. Ensure the information is up-to-date and matches the constraints.
4. SYNTHESIS: Combine the gathered information to answer the question.
5. FORMAT: 
   - Give the final answer as concise as possible.
   - Enclose the final answer in \\boxed{} notation.
   - Number format: No commas (e.g., 12345.67).
   - String format: Lowercase, no punctuation unless specified.

Question: {input}
"""


REFINE_ANSWER_PROMPT = """
Given the problem and any supporting information or data, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the problem.
2. Explain the approach and any concepts or reasoning used.
3. Show step-by-step analysis and reasoning.
4. Incorporate any supporting evidence or data into your explanation.
5. Present your final answer enclosed in \boxed{} notation, following these formatting rules:
   - If the answer is a number: don't use commas, don't use units ($ or %) unless specified
   - If the answer is a string: don't use articles, no abbreviations (e.g. for cities), write digits in plain text unless specified
   - If the answer is a comma separated list: apply the above rules based on element type

Your response should be comprehensive, well-reasoned, and easy to follow.
"""

SOLUTION_PROMPT = """
Provide a comprehensive, step-by-step solution to the given problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the key concepts and context involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. Supporting evidence, facts, or data where relevant.
6. Additional context or background information if helpful.
7. Present your final answer enclosed in \\boxed{} notation, following these formatting rules:
   - If the answer is a number: don't use commas, don't use units ($ or %) unless specified
   - If the answer is a string: don't use articles, no abbreviations (e.g. for cities), write digits in plain text unless specified
   - If the answer is a comma separated list: apply the above rules based on element type
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and well-supported.
"""


DETAILED_SOLUTION_PROMPT = """
Provide a comprehensive, step-by-step solution to the given problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the key concepts and context involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. Supporting evidence, facts, or data where relevant.
6. Additional context or background information if helpful.
7. Present your final answer enclosed in \boxed{} notation, following these formatting rules:
   - If the answer is a number: don't use commas, don't use units ($ or %) unless specified
   - If the answer is a string: don't use articles, no abbreviations (e.g. for cities), write digits in plain text unless specified
   - If the answer is a comma separated list: apply the above rules based on element type
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and well-supported.
"""
