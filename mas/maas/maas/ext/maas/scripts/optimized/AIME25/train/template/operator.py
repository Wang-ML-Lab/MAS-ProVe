import concurrent
import sys
import traceback
from typing import List
import re
import json
from tenacity import retry, stop_after_attempt, wait_fixed

from maas.ext.maas.scripts.optimized.AIME25.train.template.operator_an import *
from maas.ext.maas.scripts.optimized.AIME25.train.template.op_prompt import *
from maas.actions.action_node import ActionNode
from maas.llm import LLM
from maas.logs import logger
import asyncio



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


class Generate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

class GenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "GenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = GENERATE_COT_PROMPT.replace("{instruction}", str(instruction)).replace("{input}", str(input))
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

class MultiGenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = GENERATE_COT_PROMPT.replace("{instruction}", str(instruction)).replace("{input}", str(input))
        response1 = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        response2 = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        response3 = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        
        return {"response": [response1, response2, response3]}


class ScEnsemble(Operator):
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    def _extract_number(self, text: str) -> str:
        """Helper to standardize answer extraction for matching."""
        # 1. Try boxed
        match = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", text)
        if match:
            return match.group(1).strip()
        # 2. Fallback to last number
        numbers = re.findall(r"(?<!\d)\d{1,4}(?!\d)", text)
        return numbers[-1] if numbers else ""

    async def __call__(self, solutions: List[str], problem: str):
        # 1. Format solutions with labels (Still useful for the LLM to read distinct options)
        solution_text = ""
        for index, solution in enumerate(solutions):
            solution_text += f"Solution {chr(65 + index)}: \n{str(solution)}\n\n\n"

        # 2. Call LLM with the AIME prompt
        prompt = SC_ENSEMBLE_PROMPT.replace("{problem}", problem).replace("{solutions}", solution_text)
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

def run_code(code):
    try:
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        exec(code, global_namespace)
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    
class Programmer(Operator):
    def __init__(self, llm: LLM, name: str = "Programmer"):
        super().__init__(llm, name)

    # FIX 1: Reduce default timeout from 600s to 10s. 
    # Valid AIME solutions are instant; infinite loops are not.
    async def exec_code(self, code, timeout=10): 
        loop = asyncio.get_running_loop()
        
        # Use context manager to ensure cleanup
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        try:
            future = loop.run_in_executor(executor, run_code, code)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Force kill all processes in the executor
            for pid, process in executor._processes.items():
                try:
                    process.terminate()  # Send SIGTERM
                    process.join(timeout=1)  # Wait 1 second
                    if process.is_alive():
                        process.kill()  # Force kill with SIGKILL
                except Exception:
                    pass
            executor.shutdown(wait=False, cancel_futures=True)
            return "Error", "Code execution timed out (Infinite Loop Detected)"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"
        finally:
            # Always cleanup
            executor.shutdown(wait=False)

    async def code_generate(self, problem, analysis, feedback, mode):
        prompt = PYTHON_CODE_VERIFIER_PROMPT.replace("{problem}", problem).replace("{analysis}", analysis).replace("{feedback}", feedback)
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response

    @retry(stop=stop_after_attempt(1), wait=wait_fixed(1)) 
    async def __call__(self, problem: str, analysis: str = "None"):
        code = None
        output = None
        feedback = ""
        
        # Wrap entire function in timeout to prevent indefinite hangs
        try:
            return await asyncio.wait_for(self._execute_with_retries(problem, analysis), timeout=30)
        except asyncio.TimeoutError:
            logger.warning(f"Programmer timed out after 30s on problem")
            return {"code": code, "output": "Total execution timeout - skipping code solution"}
    
    async def _execute_with_retries(self, problem: str, analysis: str = "None"):
        code = None
        output = None
        feedback = ""
        
        # Inner loop: Try to fix the code 2 times (reduced from 3)
        for i in range(2):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            code = code_response.get("code")
            
            if not code:
                return {"code": code, "output": "No code generated"}
            
            # Execute with the new short timeout
            status, output = await self.exec_code(code)
            
            if status == "Success":
                return {"code": code, "output": output}
            else:
                # Log the short failure and try again
                print(f"Attempt {i + 1} failed: {output}")
                feedback = (
                    f"\nThe previous code failed to execute:\n"
                    f"Code: {code}\n\nError: {output}\n"
                    f"Please fix the logic errors or infinite loops."
                )
        
        return {"code": code, "output": output}
    
            
class SelfRefine(Operator):
    def __init__(self, llm: LLM, name: str = "SelfRefine"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution):
        prompt = SELFREFINE_PROMPT.replace("{problem}", problem).replace("{solution}", solution)
        response = await self._fill_node(SelfRefineOp, prompt, mode="single_fill")
        return response
    
class EarlyStop(Operator):
    def __init__(self, llm: LLM, name: str = "EarlyStop"):
        super().__init__(llm, name)

    async def __call__(self):
        return NotImplementedError
