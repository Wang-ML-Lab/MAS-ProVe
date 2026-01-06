import inspect
import re
import asyncio
from typing import Any, Callable, List, Tuple
import torch
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger
from maas.utils.sanitize import sanitize

class GAIABenchmark(BaseBenchmark):
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
        """
        Extracts the final answer from the model's response."""
        if text is None:
            return ""
        text = str(text)

        # 1. Try \boxed{...} pattern (LaTeX style)
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # 2. Try boxed{...} pattern (without backslash)
        pattern = r"boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # 3. Try "final answer is:" pattern (case insensitive)
        pattern = r"final\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)"
        final_answer_matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if final_answer_matches:
            return final_answer_matches[-1].strip()

        # 4. Try "the answer is:" pattern (case insensitive)
        pattern = r"the\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)"
        answer_matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if answer_matches:
            return answer_matches[-1].strip()

        # 5. Try <answer>...</answer> tags
        if "<answer>" in text and "</answer>" in text:
            start = text.find("<answer>") + len("<answer>")
            end = text.find("</answer>")
            return text[start:end].strip()

        # 6. Fallback: Return the last non-empty line (heuristic for simple responses)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            return lines[-1]

        return text.strip()

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = str(expected_output).strip()
        predicted_answer = self.extract_model_answer(prediction)

        if self.check_gaia_match(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def check_gaia_match(self, extracted_answer: str, expected_answer: str) -> bool:
        """
        Flexible matching logic for GAIA dataset (Numerical vs String flexible match).
        """
        try:
            normalize = lambda x: str(x).lower().strip().replace(" ", "")
            norm_resp = normalize(extracted_answer)
            norm_expected = normalize(expected_answer)
            
            # Check if expected answer is numerical (after removing commas)
            norm_expected_no_comma = norm_expected.replace(",", "")
            is_numerical = re.match(r'^-?\d+\.?\d*$', norm_expected_no_comma) is not None
            
            if is_numerical:
                # For numerical answers, remove commas and do exact match
                norm_resp_no_comma = norm_resp.replace(",", "")
                flexible_match = norm_resp_no_comma == norm_expected_no_comma
            else:
                # For string answers, allow flexible matching
                exact_match = norm_resp == norm_expected
                
                # If response has parentheses, check parenthesized text
                paren_match = re.search(r'\(([^)]+)\)', extracted_answer)
                paren_content = normalize(paren_match.group(1)) if paren_match else ""
                
                # Check if expected answer appears as a substring
                substring_match = (norm_expected in norm_resp) or (paren_content and norm_expected in paren_content)
                
                # Combined match logic for strings
                flexible_match = exact_match or substring_match
                
            return flexible_match

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
        input_text = problem.get("Question")
    
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
            logger.error(f"CRASH DETECTED for Input: {str(input_text)[:30]}...")
            traceback.print_exc()
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]