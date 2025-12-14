import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from datasets import load_dataset

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class AIME25Benchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    # async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
    #     """Override load_data to load from HuggingFace dataset"""
    #     # Load the dataset from HuggingFace
    #     ds = load_dataset("simplescaling/aime25_nofigures")
        
    #     # Convert to list of dictionaries
    #     # Assuming the dataset has 'train' split, adjust if needed
    #     split_name = list(ds.keys())[0]  # Get the first available split
    #     data = list(ds[split_name])
        
    #     if specific_indices is not None:
    #         filtered_data = [data[i] for i in specific_indices if i < len(data)]
    #         return filtered_data
    #     return data

    def extract_model_answer(self, text: str) -> int:
        """Extract numerical answer from model output (AIME answers are integers 0-999)"""
        # Look for <answer> tags first
        if "<answer>" in text and "</answer>" in text:
            answer_section = text.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            answer_section = text
        
        # Look for boxed LaTeX
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, answer_section, re.DOTALL)
        if boxed_matches:
            answer_section = boxed_matches[-1].strip()
        
        # Extract 1-3 digit numbers from answer section
        numbers = re.findall(r'\b\d{1,3}\b', answer_section)
        if numbers:
            return int(numbers[-1])
        
        # Try common patterns
        patterns = [
            r'(?:answer|result|solution) is (\d{1,3})',
            r'(?:equals|=) (\d{1,3})',
            r'therefore (\d{1,3})',
            r'thus (\d{1,3})'
        ]
        for pattern in patterns:
            match = re.search(pattern, answer_section.lower())
            if match:
                return int(match.group(1))
        
        # Fall back to any 1-3 digit number in full text
        all_numbers = re.findall(r'\b\d{1,3}\b', text)
        return int(all_numbers[-1]) if all_numbers else -1


    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        """Calculate score by comparing expected and predicted answers"""
        expected_answer = self.extract_model_answer(str(expected_output))
        predicted_answer = self.extract_model_answer(str(prediction))
        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """Check if two mathematical expressions are equal"""
        pred_str = str(prediction).strip()
        ref_str = str(reference).strip()
        
        # Try to compare as integers first (for AIME answers 0-999)
        try:
            pred_int = int(float(pred_str))
            ref_int = int(float(ref_str))
            if pred_int == ref_int:
                return True
        except:
            pass
        
        # String comparison as fallback
        if pred_str == ref_str:
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        """Check if a value can be parsed as a digit"""
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        """Parse string to float, handling percentages and commas"""
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        """Check symbolic equality using sympy"""
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        """Get source code of a function for logging"""
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        """Generate output with retry logic"""
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        """Evaluate a single problem from the dataset"""
        # AIME24 dataset fields: id, problem, answer, solution
        # We use 'problem' for the question and 'answer' for the expected numerical answer
        input_text = problem["problem"]
        expected_output = str(problem["answer"])

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return input_text, output, expected_output, uni_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        """Define the columns for the results CSV"""
        return ["question", "prediction", "expected_output", "score", "cost"]
