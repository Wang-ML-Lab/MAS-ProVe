WORKFLOW_OPTIMIZE_PROMPT = """You are building a Graph and corresponding Prompt to jointly solve {type} problems. 

CRITICAL REQUIREMENTS (MUST FOLLOW):
1. ALWAYS initialize BaseClient in __init__: self.client = BaseClient(host='127.0.0.1', port=5555)
2. EVERY operator call (Custom, Programmer, etc.) MUST include these three kwargs:
   - client=self.client
   - task_type="math"
   - question=problem
3. Example of CORRECT operator call:
   response = await self.custom(input=problem, instruction=prompt_custom.SOME_PROMPT, 
                                client=self.client, task_type="math", question=problem)
4. Example of WRONG operator call (missing kwargs - DO NOT DO THIS):
   response = await self.custom(input=problem, instruction=prompt_custom.SOME_PROMPT)

Referring to the given graph and prompt, which forms a basic example of a {type} solution approach, 
please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your 
single modification in XML tags in your reply. Ensure they are complete and correct to avoid runtime failures. When 
optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider 
Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), 
or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph 
complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical 
representation.Ensure that all the prompts required by the current graph from prompt_custom are included.Exclude any other prompts.
Output the modified graph and all the necessary Prompts in prompt_custom (if needed).
The prompt you need to generate is only the one used in `prompt_custom.XXX` within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in `prompt_custom`; please remove any unused prompts in prompt_custom.
the generated prompt must not contain any placeholders.
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It's crucial to include necessary context during the process.

IMPORTANT: The Workflow class __init__ method MUST include these properties for tool calling support:
```python
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
```
These properties MUST be included in every generated Workflow class.

MANDATORY: The Workflow class __init__ method MUST initialize BaseClient:
```python
self.client = BaseClient(host='127.0.0.1', port=5555)
```

MANDATORY: ALL operator calls MUST include these three kwargs (or they will fail at runtime):
```python
await self.custom(input=..., instruction=..., client=self.client, task_type="math", question=problem)
await self.programmer(problem=..., analysis=..., client=self.client, task_type="math", question=problem)
await self.sc_ensemble(solutions=..., problem=..., client=self.client, task_type="math", question=problem)
```
If you do not include client=self.client, task_type="math", and question=problem in EVERY operator call, the workflow will crash with "A 'client' must be provided as a kwarg" error."""


WORKFLOW_INPUT = """
Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify/ ...)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

First, provide optimization ideas. **Only one detail point can be modified at a time**, and no more than 5 lines of code may be changed per modification—extensive modifications are strictly prohibited to maintain project focus!
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
You do not need to manually import prompt_custom or operator to use them; they are already included in the execution environment.

CRITICAL PROMPT FORMAT REQUIREMENTS:
In the <prompt> section, ALL prompts MUST be valid Python code with proper variable assignments.
Format: PROMPT_NAME = \"\"\"prompt text here\"\"\"

WRONG (will cause runtime errors):
# Prompt used by Custom in the graph. Name: MY_PROMPT
# Instructions: ...

CORRECT:
MY_PROMPT = \"\"\"Instructions: ...\"\"\"

Every prompt must be a Python string variable that can be imported. Do NOT use comments to describe prompts.
"""

WORKFLOW_CUSTOM_USE = """\nHere's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n

**Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators are not yet used in the graph, try incorporating them.**
"""

WORKFLOW_TEMPLATE = """from typing import Literal, List, Dict, Callable, Optional
import {base_path}.{dataset}.workflows.template.operator as operator
import {base_path}.{dataset}.workflows.round_{round}.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from mas_proceval.clients.client_base import BaseClient

from scripts.evaluator import DatasetType

{graph}
"""
