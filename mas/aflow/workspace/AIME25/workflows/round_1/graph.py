from typing import Literal, List, Dict, Callable, Optional
import workspace.AIME25.workflows.template.operator as operator
import workspace.AIME25.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from mas_proceval.clients.client_base import BaseClient

from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.client = BaseClient(host='127.0.0.1', port=5555)
        self._tools: Optional[List[dict]] = None
        self._tool_functions: Optional[Dict[str, Callable]] = None

    @property
    def tools(self) -> Optional[List[dict]]:
        return self._tools

    @tools.setter
    def tools(self, value: List[dict]):
        self._tools = value
        self.llm.tools = value

    @property
    def tool_functions(self) -> Optional[Dict[str, Callable]]:
        return self._tool_functions

    @tool_functions.setter
    def tool_functions(self, value: Dict[str, Callable]):
        self._tool_functions = value
        self.llm.tool_functions = value

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(
            input=problem, 
            instruction="",
            client=self.client,
            task_type="math",
            question=problem
        )
        return solution['response'], self.llm.get_usage_summary()["total_cost"]