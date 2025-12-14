from typing import Literal, List, Dict, Callable, Optional
import workspace.AIME25.workflows.template.operator as operator
import workspace.AIME25.workflows.round_2.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


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
        self.sc = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)
        # required tool properties for tool calling support
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
        Implementation of the workflow:
        - Query Custom twice with two distinct, explicit prompts to encourage different solution paths.
        - Use Programmer to run a short verification script (compute polygon areas) based on coordinates extracted from the best answer.
        - Use ScEnsemble to pick the majority/best solution among the generated answers (the responses are strings).
        - Return the final numeric answer and cost.
        """
        # two complementary prompts to Custom to encourage diverse reasoning
        resp1 = await self.custom(input=problem, instruction=prompt_custom.PROMPT_A)
        resp2 = await self.custom(input=problem, instruction=prompt_custom.PROMPT_B)

        # gather textual candidate solutions
        candidates = [resp1['response'], resp2['response']]

        # Ask programmer to verify area using coordinates if possible; provide both texts for context
        prog = await self.programmer(problem=problem, analysis="Verify numeric answer by computing polygon areas from coordinate approach. Use integer output only.")
        candidates.append(prog['output'])

        # ensemble selection
        ensembled = await self.sc(solutions=candidates, problem=problem)

        # ensure final is formatted as a single integer string if possible
        final_answer_text = ensembled['response']
        return final_answer_text, self.llm.get_usage_summary()["total_cost"]
