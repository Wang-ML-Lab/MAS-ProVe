import json

operator_example = {
    "thought": {
        "Insights": "The Generate operator is designed to leverage the LLM's capability to dynamically generate responses based on input prompts. It serves as a core text generation module for agents.",
        "Overall Idea": "This operator abstracts text generation into a reusable component. By accepting an input string and optional mode, it can adapt to different prompting strategies (e.g., XML formatting) while maintaining a simple interface. The asynchronous implementation ensures compatibility with modern AI agent architectures.",
        "Implementation": "1. Inherit from base Operator class\n2. Implement async __call__ method\n3. Integrate with prompt templating\n4. Support multiple response formats through mode parameter\n5. Maintain state isolation through instance-based design"},
    "description": "Core text generation operator that uses LLMs to produce responses based on structured prompts and input data. Supports asynchronous execution and multiple response formats.",
    "code": """
    class Generate(Operator):
        def __init__(self, llm: LLM, name: str = "Generate"):
            super().__init__(llm, name)
            self.response_format = "xml"  # Default format
        
        async def __call__(self, input: str, mode: str = None):
            \"\"\"Generate response using structured prompt template
            Args:
                input: Input text or prompt template variables
                mode: Response format mode (xml/json/text)
            Returns:
                Generated response string
            \"\"\"
            prompt = GENERATE_PROMPT.format(input=input)
            response_mode = mode or self.response_format
            response = await self._fill_node(GenerateOp, prompt, mode=response_mode)
            return self._postprocess(response)
        
        def _postprocess(self, raw_response: str) -> str:
            \"\"\"Clean and validate generated response\"\"\"
            return raw_response.strip().replace("\\n\\n", "\\n")
"""
}

Generate = {
    "thought": {
        "Insights": "The Generate operator is designed to leverage the LLM's capability to dynamically generate responses based on input prompts. It serves as a core text generation module for agents.",
        "Overall Idea": "This operator abstracts text generation into a reusable component. By accepting an input string and optional mode, it can adapt to different prompting strategies (e.g., XML formatting) while maintaining a simple interface. The asynchronous implementation ensures compatibility with modern AI agent architectures.",
        "Implementation": "1. Inherit from base Operator class\n2. Implement async __call__ method\n3. Integrate with prompt templating\n4. Support multiple response formats through mode parameter\n5. Maintain state isolation through instance-based design"},
    "description": "Core text generation operator that uses LLMs to produce responses based on structured prompts and input data. Supports asynchronous execution and multiple response formats.",
    "code": """
    class Generate(Operator):
        def __init__(self, llm: LLM, name: str = "Generate"):
            super().__init__(llm, name)
            self.response_format = "xml"  # Default format
        
        async def __call__(self, input: str, mode: str = None):
            \"\"\"Generate response using structured prompt template
            Args:
                input: Input text or prompt template variables
                mode: Response format mode (xml/json/text)
            Returns:
                Generated response string
            \"\"\"
            prompt = GENERATE_PROMPT.format(input=input)
            response_mode = mode or self.response_format
            response = await self._fill_node(GenerateOp, prompt, mode=response_mode)
            return self._postprocess(response)
        
        def _postprocess(self, raw_response: str) -> str:
            \"\"\"Clean and validate generated response\"\"\"
            return raw_response.strip().replace("\\n\\n", "\\n")
"""
}

GenerateCoT = {
    "thought": {
        "Insights": "Enhances reasoning transparency through explicit chain-of-thought generation",
        "Overall Idea": "Structured CoT generation for complex problem solving with traceable logic steps",
        "Implementation": "1. Specialized CoT prompt template\n2. XML response parsing\n3. Async-compatible execution"
    },
    "description": "Chain-of-Thought generator that produces explicit reasoning steps using structured XML output format",
    "code": """
    class GenerateCoT(Operator):
        def __init__(self, llm: LLM, name: str = "GenerateCoT"):
            super().__init__(llm, name)
            self.cot_cache = {}

        async def __call__(self, input: str, mode: str = None):
            \"\"\"Generate structured reasoning chain for complex problems
            Args:
                input: Problem description with solution requirements
                mode: Output structure format (default: XML)
            Returns:
                Dictionary with 'reasoning_chain' and 'final_answer' keys
            \"\"\"
            prompt = GENERATECOT_PROMPT.format(input=input)
            response = await self._fill_node(GenerateCoTOp, prompt, mode="xml_fill")
            return self._parse_cot(response)

        def _parse_cot(self, raw_response: dict) -> dict:
            \"\"\"Extract reasoning steps from XML response\"\"\"
            return {
                'reasoning_chain': raw_response.get('steps', []),
                'final_answer': raw_response.get('conclusion', '')
            }
"""
}

MultiGenerateCoT = {
    "thought": {
        "Insights": "Diverse reasoning paths improve solution robustness",
        "Overall Idea": "Parallel generation of multiple distinct reasoning chains",
        "Implementation": "1. Concurrent CoT generation\n2. Result aggregation\n3. Diversity sampling"
    },
    "description": "Generates multiple independent reasoning chains for solution verification and ensemble analysis",
    "code": """
    class MultiGenerateCoT(Operator):
        def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
            super().__init__(llm, name)
            self.max_attempts = 3

        async def __call__(self, input: str, mode: str = None):
            \"\"\"Produce multiple reasoning paths for same input problem
            Args:
                input: Target problem statement
                mode: Consistency mode for parallel generations
            Returns:
                List of dictionaries with varied reasoning approaches
            \"\"\"
            prompt = GENERATECOT_PROMPT.format(input=input)
            tasks = [self._fill_node(GenerateCoTOp, prompt, mode="xml_fill") 
                    for _ in range(3)]
            results = await asyncio.gather(*tasks)
            return [self._parse_cot(r) for r in results]
"""
}

ScEnsemble = {
    "thought": {
        "Insights": "AIME problems require strict integer constraints; letter-based voting is error-prone if the model outputs the value directly.",
        "Overall Idea": "Consensus mechanism that selects the target integer value via LLM, then programmatically retrieves the matching full solution.",
        "Implementation": "1. Heuristic number extraction (boxed/last integer)\n2. LLM selects target integer\n3. Reverse-lookup to find matching solution text"
    },
    "description": "Ensemble operator that identifies the best solution by selecting a consensus integer answer and finding the corresponding reasoning trace via regex matching.",
    "code": """
    class ScEnsemble(Operator):
        def __init__(self, llm: LLM, name: str = "ScEnsemble"):
            super().__init__(llm, name)

        def _extract_number(self, text: str) -> str:
            \"\"\"Helper to standardize answer extraction for matching.\"\"\"
            # 1. Try boxed
            match = re.search(r"\\\\boxed{((?:[^{}]|{[^{}]*})*)}", text)
            if match:
                return match.group(1).strip()
            # 2. Fallback to last number
            numbers = re.findall(r"(?<!\d)\d{1,4}(?!\d)", text)
            return numbers[-1] if numbers else ""

        async def __call__(self, solutions: List[str], problem: str):
            # 1. Format solutions with labels (Still useful for the LLM to read distinct options)
            solution_text = ""
            for index, solution in enumerate(solutions):
                solution_text += f"Solution {chr(65 + index)}: \\n{str(solution)}\\n\\n\\n"

            # 2. Call LLM with the AIME prompt
            prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
            response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

            # 3. Get the Target Integer (e.g., "113")
            # Changed from "solution_letter" to "final_answer" as requested
            target_answer = str(response.get("final_answer", "")).strip()

            # 4. Logic Fix: Find the solution text that matches this target integer
            # We cannot use 'answer_mapping' because target_answer is '113', not 'A'.
            selected_solution = solutions[0] if solutions else ""

            for sol in solutions:
                # Check if this solution produced the target answer
                if self._extract_number(sol) == target_answer:
                    selected_solution = sol
                    break
            
            return {"response": selected_solution}
"""
}

Programmer = {
    "thought": {
        "Insights": "Combines code generation with safe execution validation",
        "Overall Idea": "Automated programming workflow with execution feedback loop",
        "Implementation": "1. Code generation with error context\n2. Sandboxed execution\n3. Adaptive retry mechanism"
    },
    "description": "Self-debugging code generator that iteratively improves solutions through execution feedback and retries",
    "code": """
    class Programmer(Operator):
        def __init__(self, llm: LLM, name: str = "Programmer"):
            super().__init__(llm, name)
            self.execution_history = []

        async def _safe_execute(self, code: str, timeout: int = 30) -> tuple:
            \"\"\"Execute code in isolated environment with resource limits\"\"\"
            try:
                loop = asyncio.get_running_loop()
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(
                        executor, 
                        self._run_in_sandbox,  # Assume sandbox execution method
                        code,
                        timeout
                    )
                    return await asyncio.wait_for(future, timeout=timeout+5)
            except Exception as e:
                return ("Runtime Error", str(e))

        async def code_generate(self, problem: str, context: str, feedback: str) -> dict:
            \"\"\"Generate code with error context integration\"\"\"
            prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
                problem=problem,
                analysis=context,
                feedback=feedback
            )
            return await self._fill_node(CodeGenerateOp, prompt, "code_fill", function_name="solve")

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        async def __call__(self, problem: str, context: str = None):
            \"\"\"Iterative code generation-execution loop
            Args:
                problem: Programming problem description
                context: Initial analysis or specifications
            Returns:
                Dictionary with final code and execution results
            \"\"\"
            feedback = ""
            for attempt in range(3):
                # Generate code with accumulated feedback
                response = await self.code_generate(problem, context, feedback)
                code = response.get("code", "")
                
                if not code:
                    return {"status": "generation_failed", "code": code}
                
                # Execute and analyze results
                status, output = await self._safe_execute(code)
                self.execution_history.append({
                    "attempt": attempt+1,
                    "code": code,
                    "status": status,
                    "output": output
                })
                
                if status == "Success":
                    return {"status": "success", "code": code, "output": output}
                
                # Build feedback for next iteration
                feedback = (f"\\nExecution attempt {attempt+1} failed:\\n"
                        f"Status: {status}\\n"
                        f"Error output: {output[:500]}\\n"
                        f"Relevant code snippet: {code.split('\\n')[-5:]}")
            
            return {
                "status": "max_retries_exceeded",
                "code": self.execution_history[-1]["code"],
                "errors": [e["output"] for e in self.execution_history]
            }
"""
}

SelfRefine = {
    "thought": {
        "Insights": "Enables iterative self-improvement of solutions through reflection",
        "Overall Idea": "Automated solution refinement using error feedback loops",
        "Implementation": "1. Error analysis prompt template\n2. Solution regeneration workflow\n3. Version tracking"
    },
    "description": "Iterative solution refiner that analyzes execution feedback and produces improved solution variants",
    "code": """
    class SelfRefine(Operator):
        def __init__(self, llm: LLM, name: str = "SelfRefine"):
            super().__init__(llm, name)
            self.version_control = []

        async def __call__(self, problem, solution, mode: str = None):
            \"\"\"Refine solution through critical self-analysis
            Args:
                problem: Original problem statement
                solution: Current solution version
                mode: Refinement strategy (strict/balanced/creative)
            Returns:
                Dictionary containing refined solution and revision notes
            \"\"\"
            prompt = SELFREFINE_PROMPT.format(
                problem=problem, 
                solution=solution,
                history="\\n".join(self.version_control[-3:])
            )
            response = await self._fill_node(SelfRefineOp, prompt, mode="xml_fill")
            self.version_control.append(f"Version {len(self.version_control)+1}: {response['revision_summary']}")
            return {
                "refined_solution": response["improved_solution"],
                "revision_notes": response["critical_analysis"]
            }
"""
}

EarlyStop = {
    "thought": {
        "Insights": "Prevents unnecessary computation through progress monitoring",
        "Overall Idea": "Conditional termination based on convergence metrics",
        "Implementation": "1. Performance plateau detection\n2. Resource usage monitoring\n3. Validation metric tracking"
    },
    "description": "Conditional termination operator that halts processing when quality plateaus or resource limits approach",
    "code": """
    class EarlyStop(Operator):
        def __init__(self, llm: LLM, name: str = "EarlyStop"):
            super().__init__(llm, name)
            self.metric_window = []
            self.stagnation_threshold = 3

        async def __call__(self, metrics: dict, resource_usage: dict):
            \"\"\"Evaluate stopping conditions based on performance trends
            Args:
                metrics: Current iteration quality metrics
                resource_usage: System resource consumption data
            Returns:
                Boolean indicating whether to terminate process
            \"\"\"
            self.metric_window.append(metrics["score"])
            if len(self.metric_window) > self.stagnation_threshold:
                recent_gain = abs(np.mean(self.metric_window[-3:]) - np.mean(self.metric_window[-6:-3]))
                if recent_gain < 0.01:
                    return True
            
            if resource_usage.get("memory_percent", 0) > 90:
                return True
                
            if metrics.get("improvement_rate", 0) < 0.05:
                return True
                
            return False
"""
}

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """
# Overview
You are an expert machine learning researcher specializing in designing agentic systems. Your objective is to create building blocks such as reasoning modules and problem-solving workflows within these systems to solve complex mathematical tasks. Specifically, you aim to design an optimal agent that performs exceptionally on the AIME benchmark (American Invitational Mathematics Examination). The AIME evaluates exceptional mathematical reasoning capabilities, serving as the intermediate examination between the AMC 10/12 and the USAMO. Each problem includes:
- A challenging natural language description requiring creative problem-solving and deep insight
- A strict format constraint: The final answer is always an integer between 000 and 999 (inclusive)
- Multi-step reasoning that often combines multiple mathematical disciplines (algebra, geometry, number theory, combinatorics)

# Example Question from AIME
{"problem": "The sum of four consecutive terms of an arithmetic sequence is 120. The sum of the squares of these terms is 3680. Find the absolute value of the difference between the first and last terms.", "solution": "Let the four consecutive terms be $a-3d, a-d, a+d, a+3d$, where the common difference is $2d$. The sum of the terms is $4a = 120$, so $a = 30$. The sum of the squares is $(a-3d)^2 + (a-d)^2 + (a+d)^2 + (a+3d)^2 = 4a^2 + 20d^2$. Substituting $a=30$, we have $4(900) + 20d^2 = 3600 + 20d^2 = 3680$. This simplifies to $20d^2 = 80$, so $d^2 = 4$, which implies $d = 2$ (taking positive $d$ for the difference magnitude). The terms are $24, 28, 32, 36$. The difference between the first and last terms is $|(a+3d) - (a-3d)| = |6d| = 6(2) = 12$. Thus, the answer is \\boxed{12}."}
# Operator code template:
class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()

class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class Generate(Operator):
    GENERATE_PROMPT = '''
You are tasked with solving the following Python programming problem. Generate a complete, syntactically correct Python function that strictly adheres to the given requirements.

Problem:
{input}

Follow these steps:
1. Analyze the problem requirements and identify edge cases
2. Design a solution that passes all implied test cases
3. Implement the function with clear variable names and comments

Ensure:
- The code directly implements the requested functionality
- All parameters and return types match the problem specification
- Exception handling for edge cases is included when necessary '''

    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, mode: str = None):
        prompt = self.GENERATE_PROMPT.format(input=input)
        response = await self._fill_node(GenerateOp, prompt, mode="xml_fill")
        return response

# Discovered architecture archive
Here is the archive of the discovered operator architectures:
[ARCHIVE]

# Output Instruction and Example:
The output should be a JSON object with the following structure.The first key should be (“thought”), and it should capture your thought process for designing the next operator. The second key (“description”) corresponds to the brief description of your next operator. Finally, the last key (“code”) corresponds to the exact operator and its prompt in Python code that you would like to try. You must write COMPLETE CODE in “code”: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

- thought: Captures your thought process for designing the next operator.
  - Reason about what the next interesting operator should be.
  - Describe your reasoning and the overall concept behind the operator design.
  - Detail the implementation steps.
- description: A brief description of your next operator.
- code: The exact operator and its prompt in Python code. Ensure the code is complete, reliable, and reusable.


Here is an example of the output format for the next operator: 
[operator_example]

You must strictly follow the exact input/output interface used above. Also, it could be helpful to set the LLM’s role and temperature to further control the LLMs response. DON'T try to use some function that doesn't exist. In __call__(), you need to specify the instruction, input information, the prompt and the required output fields class for operators to do their specific part of the architecture. 

# Your task 
You are highly proficient in prompting techniques and well-versed with agentic systems from academic literature. Your goal is to maximize performance metrics by proposing innovative and effective new operators.
Instructions:
1. Analyze the Discovered Operators: Carefully review the operators in the archive to identify strengths, weaknesses, and areas for improvement.
2. Draw Insights: Extract lessons and insights from existing operators to inform the design of the next operator.
3. Innovate: Think creatively to design an operator that addresses current limitations or explores new functionalities, drawing inspiration from related agent papers or other research areas.
4. Design the Operator: Propose the next operator's `thought`, `description`, and `code` following the specified format.
5. Ensure Completeness: The generated code must be complete, reliable, and reusable, fitting seamlessly into the existing architecture.

Execution Steps:
1. Generate Output: Produce the `thought`, `description`, and `code` fields for the new operator, ensuring adherence to the guidelines.
2. Validate Output: Ensure the generated JSON is correctly formatted and the code is syntactically and functionally correct.

THINK OUTSIDE THE BOX and leverage interdisciplinary insights to enhance the agentic system's capabilities.
"""

def get_prompt_aime(current_archive):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt .replace("[operator_example]", json.dumps(operator_example))

    return system_prompt, prompt

def get_init_archive_aime():
    return [Generate, GenerateCoT, MultiGenerateCoT, ScEnsemble, Programmer, SelfRefine, EarlyStop]
