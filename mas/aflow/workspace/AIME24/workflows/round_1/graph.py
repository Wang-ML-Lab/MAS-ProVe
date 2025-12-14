from typing import Literal
import workspace.AIME24.workflows.template.operator as operator
import workspace.AIME24.workflows.round_1.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction="", client=self.client, task_type="math", question=problem)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]