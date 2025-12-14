from typing import Literal, List, Dict, Callable, Optional
import workspace.AIME25.workflows.template.operator as operator
import workspace.AIME25.workflows.round_3.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        from typing import Optional, List, Dict, Callable
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # required tool properties
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

        # operators
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow:
        - Run two diverse custom prompts to produce two candidate solutions.
        - Run Programmer to attempt a code-based verification/computation using combined analysis.
        - Use ScEnsemble to aggregate the two custom responses and the programmer output into a final response.
        - Return final response and cost.
        """
        # First careful prompt: structured step-by-step reasoning with concise final answer
        resp_a = await self.custom(input=problem, instruction=prompt_custom.PROMPT_A)
        # Second careful prompt: self-consistency style: enumerate approach & count possibilities where applicable
        resp_b = await self.custom(input=problem, instruction=prompt_custom.PROMPT_B)

        # Use programmer to verify and compute exact numeric answer when possible
        prog_input = "Problem: " + problem + "\nResponseA: " + resp_a['response'] + "\nResponseB: " + resp_b['response']
        prog = await self.programmer(problem=prog_input, analysis="Use Python to compute or verify a single integer final answer. Output only the final numeric answer in 'output'.")

        # Prepare list of solution strings for ensemble (use outputs)
        solutions = [resp_a.get('response', '').strip(), resp_b.get('response', '').strip(), prog.get('output', '').strip()]
        ensemble = await self.sc_ensemble(solutions=solutions, problem=problem)

        return ensemble['response'], self.llm.get_usage_summary()["total_cost"]
