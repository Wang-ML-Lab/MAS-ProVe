import re
import string
from typing import Callable, List, Tuple
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger

class GPQABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> str:
        """
        Normalize answer for evaluation by:
        1. Converting to lowercase
        2. Removing parentheses, brackets around options
        3. Removing whitespace
        """
        # Remove various forms of option markers: (A), [A], A), A.
        s = re.sub(r'[\(\[\{]([A-Za-z])[\)\]\}]|([A-Za-z])[\.:\)]', r'\1\2', s)
        return s.lower().strip()

    def extract_choice(self, text: str) -> str:
        """
        Extract final multiple-choice option (A/B/C/D) from free-form model output.
        Falls back to normalized text if no option marker is found.
        """
        if text is None:
            return ""

        s = str(text)

        # Fast path for already-clean labels like "A".
        s_strip = s.strip()
        if re.fullmatch(r"[A-Da-d]", s_strip):
            return s_strip.upper()

        # Common patterns in generated responses, keep the last occurrence.
        patterns = [
            r"(?i)\banswer\s*[:\-]?\s*([A-D])\b",
            r"(?i)\boption\s*[:\-]?\s*([A-D])\b",
            r"\(([A-Da-d])\)",
            r"\b([A-Da-d])[\.:\)]\b",
            r"\b([A-Da-d])\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, s)
            if matches:
                return matches[-1].upper()

        # Fallback: normalized text (keeps previous behavior as last resort)
        return self.normalize_answer(s)

    def extract_options(self, question_text: str) -> dict:
        """
        Extract multiple-choice options from a GPQA question stem.
        Returns a mapping like {'A': '...', 'B': '...', ...}.
        """
        if not question_text:
            return {}

        pattern = re.compile(r"(?:^|\n)\s*([A-D])\.\s*(.*?)(?=(?:\n\s*[A-D]\.\s*)|\Z)", re.DOTALL)
        options = {}
        for match in pattern.finditer(str(question_text)):
            letter = match.group(1).upper()
            option_text = match.group(2).strip()
            options[letter] = option_text
        return options

    def match_prediction_to_choice(self, prediction: str, question_text: str) -> str:
        """
        Convert a free-form prediction into one of A/B/C/D by matching either:
        - an explicit choice letter in the prediction, or
        - the prediction text against the option texts in the question.
        """
        pred_choice = self.extract_choice(prediction)
        if pred_choice in {"A", "B", "C", "D"}:
            return pred_choice

        options = self.extract_options(question_text)
        if not options:
            return pred_choice

        pred_norm = self.normalize_answer(prediction)

        # Try exact / substring matches against the option texts.
        for letter, option_text in options.items():
            option_norm = self.normalize_answer(option_text)
            if pred_norm == option_norm or pred_norm in option_norm or option_norm in pred_norm:
                return letter

        return pred_choice

    def calculate_score(self, ground_truth: str, prediction: str, question_text: str = "") -> Tuple[float, str]:
        """
        Compute exact match score between prediction and ground truth answers.
        Score is 1.0 if strings match exactly after normalization, 0.0 otherwise.
        """
        gt_choice = self.extract_choice(ground_truth)
        pred_choice = self.match_prediction_to_choice(prediction, question_text)
        score = 1.0 if pred_choice == gt_choice else 0.0
        return score, pred_choice

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["question"]
        expected_output = problem["answer"]
        inputs = input_text

        try:
            output, cost = await self._generate_output(graph, inputs)
            score, extracted_output = self.calculate_score(expected_output, output, input_text)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]