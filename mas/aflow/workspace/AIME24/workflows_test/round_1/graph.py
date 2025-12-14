from typing import Literal, List, Dict, Callable, Optional
import workspace.AIME24.workflows.template.operator as operator
import workspace.AIME24.workflows.round_3.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from mas_proceval.clients.client_base import BaseClient

from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: "DatasetType",
    ) -> None:
        from typing import Optional, List, Dict, Callable
        # required properties for tool support
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

        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        # MANDATORY client init
        self.client = BaseClient(host='127.0.0.1', port=5555)

    async def __call__(self, problem: str):
        """
        Pipeline:
         1) Custom: produce structured analysis and plan
         2) Programmer: perform rigorous computation / enumeration when needed
         3) ScEnsemble: verify / vote among candidate final answers
        """
        # 1) Broad, structured analysis to decompose the problem and propose approaches
        analysis_resp = await self.custom(
            input=problem,
            instruction=prompt_custom.ANALYSIS_PROMPT,
            client=self.client, task_type="math", question=problem
        )

        # 2) Use Programmer to run exact computations when analysis suggests enumeration or modular arithmetic
        prog_resp = await self.programmer(
            problem=problem,
            analysis=analysis_resp['response'],
            client=self.client, task_type="math", question=problem
        )

        # 3) Ensemble: combine and verify outputs (analysis text, programmer numeric result)
        # Prepare candidate solutions: prefer numeric outputs from programmer, fallback to numbers extracted from analysis_resp
        candidates = []
        if prog_resp.get("output"):
            candidates.append(prog_resp["output"].strip())
        if analysis_resp.get("response"):
            candidates.append(analysis_resp["response"].strip())
        # ensure at least one candidate
        if not candidates:
            candidates = ["0"]

        ensemble_resp = await self.sc_ensemble(
            solutions=candidates,
            problem=problem,
            client=self.client, task_type="math", question=problem
        )

        # Return ensemble response and cost summary
        return ensemble_resp['response'], self.llm.get_usage_summary()["total_cost"]
