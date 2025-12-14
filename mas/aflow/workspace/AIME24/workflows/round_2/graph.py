from typing import Literal, List, Dict, Callable, Optional
import workspace.AIME24.workflows.template.operator as operator
import workspace.AIME24.workflows.round_2.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.client = BaseClient(host='127.0.0.1', port=5555)
        # Required tool integration properties
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
        Pipeline:
        1) Use Custom with a careful reasoning prompt to produce structured analysis and a candidate short answer.
        2) Use Programmer to perform exact symbolic or numeric verification when appropriate, requesting code and output.
        3) Use ScEnsemble to reconcile multiple candidate answers (Custom candidate + Programmer output) and return final answer and cost.
        """
        # 1) analysis and candidate from Custom
        custom_resp = await self.custom(
            input=problem,
            instruction=prompt_custom.MATH_ANALYSIS_PROMPT,
            client=self.client,
            task_type="math",
            question=problem
        )
        candidate = custom_resp['response']

        # 2) try to get exact verification from Programmer (ask to produce minimal code that returns final numeric answer)
        prog_resp = await self.programmer(
            problem=problem,
            analysis=f"Verify and compute the final numeric answer for the problem. Use the candidate answer: {candidate}",
            client=self.client,
            task_type="math",
            question=problem
        )
        prog_output = prog_resp.get('output', '').strip()
        prog_code = prog_resp.get('code', '').strip()

        # 3) assemble candidate solutions and run ensemble verification
        solutions = []
        if candidate:
            solutions.append(candidate)
        if prog_output:
            solutions.append(prog_output)

        # if programmer produced code but no clean output, prefer to run a short evaluator via the programmer output
        if not solutions and prog_code:
            # ask programmer again to return only the computed numeric answer
            prog_resp2 = await self.programmer(
                problem=problem,
                analysis="Run the code and output only the final numeric answer (one number). If symbolic, give exact rational as m/n.",
                client=self.client,
                task_type="math",
                question=problem
            )
            if prog_resp2.get('output'):
                solutions.append(prog_resp2['output'])

        # if still no solutions, fallback to candidate (may be empty string)
        if not solutions:
            solutions = [candidate if candidate else ""]

        ensemble = await self.sc_ensemble(
            solutions=solutions,
            problem=problem,
            client=self.client,
            task_type="math",
            question=problem
        )
        final = ensemble.get('response', solutions[0])

        # return final response and cost as required
        return final, self.llm.get_usage_summary()["total_cost"]
