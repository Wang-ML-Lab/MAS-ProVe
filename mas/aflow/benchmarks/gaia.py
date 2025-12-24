# -*- coding: utf-8 -*-
# @Date    :
# @Author  : all
# @Desc    : test on GAIA benchmark
import re
from typing import Callable, List, Tuple

from datasets import load_dataset
from ddgs import DDGS
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class GAIABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str, subset: str = "test"):
        super().__init__(name, file_path, log_path)
        self.subset = subset
        self.dataset = None
        
        # Define web search tool
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Perform a web search using DuckDuckGo to find information on the internet. Use this when you need to look up current information, facts, or details that you don't know.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "The search query to look up on the web",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of search results to return (default: 5)",
                                "default": 5
                            },
                        },
                        "required": ["search_query"],
                    },
                }
            }
        ]
        
        # Define tool functions mapping
        self.tool_functions = {
            "web_search": self.web_search
        }
    
    def web_search(self, search_query: str, max_results: int = 5):
        """Perform web search and return formatted results"""
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))
        
        formatted_output = "Search results:\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            link = result.get('href', '')
            snippet = result.get('body', 'No content available.')
            formatted_output += f"--- SOURCE {i}: {title} ---\nURL: {link}\n\nCONTENT:\n{snippet}\n\n"
        
        return formatted_output

    # async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
    #     """
    #     Override base class load_data to load GAIA dataset from JSON file.
    #     """
    #     data_files = {"test": "data/datasets/GAIA.json"}
    #     self.dataset = load_dataset("json", data_files=data_files, split=self.subset)
    #     print(f"Loaded {len(self.dataset)} examples")
        
    #     # Convert dataset to list of dictionaries
    #     data = [dict(example) for example in self.dataset]
        
    #     if specific_indices is not None:
    #         filtered_data = [data[i] for i in specific_indices if i < len(data)]
    #         return filtered_data
        
    #     return data

    def extract_answer(self, response: str) -> str:
    # Try <answer> tags first
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            return response[start:end].strip()
        
        # Fallback: look for "Answer: X" pattern at the end
        import re
        answer_match = re.search(r'Answer:\s*(.+?)(?:\s*\(.*?\))?\s*\.?\s*$', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Last resort: return the whole response
        return response.strip()

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by converting to lowercase, stripping whitespace, and removing spaces.
        """
        return str(text).lower().strip().replace(" ", "")

    def calculate_score(self, expected_answer: str, response: str) -> Tuple[float, str]:
        """
        Calculate score for GAIA benchmark with flexible matching logic.
        Handles both numerical and string answers appropriately.
        """
        extracted_answer = self.extract_answer(response)
        
        norm_resp = self.normalize_text(extracted_answer)
        norm_expected = self.normalize_text(expected_answer)
        
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
            paren_content = self.normalize_text(paren_match.group(1)) if paren_match else ""
            
            # Check if expected answer appears as a substring (helps for 'Peng Li (Li Peng)')
            substring_match = norm_expected in norm_resp or norm_expected in paren_content
            
            # Combined match logic for strings
            flexible_match = exact_match or substring_match
        
        score = 1.0 if flexible_match else 0.0
        return score, extracted_answer

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        # Check if graph has tools attribute and set it for GAIA
        # print(f"[GAIA] Checking if graph has tools attribute: {hasattr(graph, 'tools')}")
        if hasattr(graph, 'tools'):
            # print(f"[GAIA] Setting {len(self.tools)} tool(s) on graph")
            graph.tools = self.tools
            graph.tool_functions = self.tool_functions
        # else:
            # print(f"[GAIA] Graph does not have tools attribute. Graph type: {type(graph)}")
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        """
        Evaluate a single GAIA problem.
        
        Args:
            problem: Dictionary containing 'Question' and 'Final answer' keys
            graph: The agent/model callable to generate predictions
            
        Returns:
            Tuple of (question, prediction, expected_answer, score, cost)
        """
        input_text = problem["Question"]
        expected_answer = problem["answer"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted_answer = self.calculate_score(expected_answer, output)

            if score == 0:
                self.log_mismatch(input_text, expected_answer, output, extracted_answer)

            return input_text, output, expected_answer, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_answer, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_answer", "score", "cost"]
