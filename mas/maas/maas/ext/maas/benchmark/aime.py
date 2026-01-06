import inspect
import re
import asyncio
from typing import Any, Callable, List, Tuple
import torch
import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger
from maas.utils.sanitize import sanitize

class AIMEBenchmark(BaseBenchmark):
    def __init__(self, 
                 name: str,
                 file_path: str,
                 log_path: str,
                 batch_size: int,
                 controller: torch.nn.Module,
                 operator_embeddings: List[List[float]],
                 optimizer: torch.optim.Optimizer):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

    def extract_model_answer(self, text: str) -> str:
        # 1. Standard Boxed extraction (Preferred)
        if text is None:
            return ""
        text = str(text)
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # 2. Fallback: Heuristic for AIME (often just a number at the end)
        # We look for the last standalone number if no box is found
        numbers = re.findall(r"(?<!\d)\d{1,4}(?!\d)", text)
        if numbers:
            return numbers[-1]
            
        return ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        try:
            # 1. Extract numbers from strings if needed
            def get_val(s):
                s = str(s).strip()
                # Remove \boxed{} if present
                if "\\boxed" in s:
                    s = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", s).group(1)
                # Remove non-numeric chars except dot and minus
                s = re.sub(r"[^\d.-]", "", s)
                return float(s)

            return get_val(prediction) == get_val(reference)
        except Exception:
            return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        # Increased timeout: 3 parallel runs × ~300s each = need ~900s total
        return await asyncio.wait_for(graph(input_text), timeout=1200)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        # Ensure your dataset keys match these
        # print(f"DEBUG DATASET KEYS: {problem.keys()}")
        input_text = problem.get("problem")
        expected_output = problem.get("answer")
        try:
            output, cost, logprob = await self._generate_output(graph, input_text)
            if not output:
                raise ValueError("output is empty")

            uni_score, extracted_output = self.calculate_score(expected_output, output)
            
            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return input_text, output, expected_output, uni_score, cost, logprob

        except Exception as e:
            import traceback
            logger.error(f"CRASH DETECTED for Input: {input_text[:30]}...")
            traceback.print_exc()
            # logger.info(f"Error processing sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]