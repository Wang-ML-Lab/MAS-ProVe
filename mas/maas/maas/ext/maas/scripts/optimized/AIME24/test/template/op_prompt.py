SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the numerical answer that appears most frequently or consistently across them.
CRITICAL AIME CONSTRAINT: The correct answer must be an integer between 0 and 999 (inclusive). 

In the "thought" field, provide a detailed explanation of your thought process. Compare the final values from the different solutions. If there is a tie or conflict, prioritize the solution with the most rigorous derivation that adheres to AIME constraints.

In the "final_answer" field, output ONLY the single integer value (e.g., 42, 105, 0). Do not include latex formatting, boxing, or text.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function must return the final calculation result.
4. Ensure precision is handled correctly (e.g., use `fractions.Fraction` or integer arithmetic where possible to avoid floating point errors).

AIME SPECIFIC: The final return value should ideally be an integer between 0 and 999. If the math results in a float like `12.0`, cast it to an integer `12`.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""

SELFREFINE_PROMPT = """
You are an assistant specialized in refining solutions to problems.

Problem:
{problem}

Solution:
{solution}

Instruction:
Analyze the above solution for any errors or suboptimal aspects. 
1. Verification: Does the mathematical logic hold up under scrutiny?
2. Format Check: Is the final answer an integer between 000 and 999? If the answer is a fraction, negative, or large number, re-evaluate the steps as this violates AIME rules (unless the problem asks for a component of the answer, like 'm+n').
3. Refinement: Make iterative improvements to enhance correctness and efficiency. Provide the refined solution below.
"""

GENERATE_COT_PROMPT = '''
Mathematical Reasoning Instruction
{instruction}

Current Problem: {input}

Demonstration Examples (AIME style):

1. Problem: Let $x$ and $y$ be positive integers such that $x^2 + y^2 = 122$. Find the value of $x+y$ given that $x > y$.
   Analysis:
   We look for perfect squares summing to 122.
   Squares less than 122: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121.
   Check complements:
   $122 - 1 = 121$ ($11^2$). Pairs: $(1, 11)$.
   $122 - 4 = 118$ (not square).
   Since $x > y$, we have $x=11$ and $y=1$.
   Calculate sum: $x+y = 11+1 = 12$.
   \boxed{12}

2. Problem: Find the number of integers $n$ with $1 \le n \le 100$ such that $n^3 - n$ is divisible by 12.
   Analysis:
   Factor expression: $n^3 - n = (n-1)n(n+1)$.
   This is the product of three consecutive integers, so it is always divisible by 3.
   We need divisibility by 4.
   Case 1: $n$ is odd. Then $n-1$ and $n+1$ are consecutive even numbers. One is divisible by 2, the other by 4. Product is divisible by 8 (and thus 4). valid.
   Case 2: $n$ is even. Then $n-1, n+1$ are odd. $n$ must be divisible by 4.
   Count:
   - Odd numbers in 1-100: 50.
   - Multiples of 4 in 1-100: 25.
   Total = 50 + 25 = 75.
   \boxed{75}

Solution Protocol:
1. Parse problem statement carefully, noting AIME constraints (answer is integer 0-999).
2. Perform stepwise symbolic derivation.
3. Verify intermediate results and ensure the final answer is a valid integer.
4. Present final answer in boxed notation.

Step-by-Step Analysis:
'''