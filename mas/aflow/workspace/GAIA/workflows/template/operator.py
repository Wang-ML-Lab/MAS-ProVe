import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.GAIA.workflows.template.operator_an import *
from workspace.GAIA.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import re
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator


from scripts.operators import Operator


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    @llm_parallel_search_decorator
    async def __call__(self, input, instruction, **kwargs):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    @llm_parallel_search_decorator
    async def __call__(self, input: str, mode: str = None, **kwargs) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response

class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    @llm_parallel_search_decorator
    async def __call__(self, solutions: List[str], **kwargs):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}