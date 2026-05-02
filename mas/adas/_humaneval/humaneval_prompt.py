import json




EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting HumanEval agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the coding agent design.\n**Implementation:**\ndescribe the implementation step by step, focusing on generating, debugging, and refining Python code.",
    "name": "Code Planner",
    "code": """async def forward(self, taskInfo):
    coding_instruction = "Read the function signature, docstring, and tests carefully. Think step by step, then return a complete Python implementation only."
    coding_agent = LLMAgentBase(['thinking', 'answer'], 'Code Planner Agent')
    thinking, answer = await coding_agent([taskInfo], coding_instruction)
    return answer
"""
}

COT = {
    "thought": "For HumanEval, chain-of-thought should help the model reason about the function signature, edge cases, and test behavior before writing code. The key idea is to separate planning from implementation so the final output is a complete function body that passes the hidden tests.",
    "name": "Code-Then-Implement",
    "code": """async def forward(self, taskInfo):
    cot_instruction = "Analyze the function signature, docstring, and examples. Think step by step about corner cases, then write the full Python implementation. Return code only in the implementation field."

    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Code-Then-Implement Agent')

    cot_agent_inputs = [taskInfo]

    thinking, answer = await cot_agent(cot_agent_inputs, cot_instruction)

    return answer
"""
}

COT_SC = {"thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.",
          "name": "Self-Consistency with Code",
          "code": """async def forward(self, taskInfo):
    cot_instruction = "Think step by step about the task, then produce a complete Python implementation. Focus on correctness, edge cases, and matching the tests."
    N = 5 # Number of code-generation samples

    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Self-Consistency Coding Agent', temperature=0.8) for _ in range(N)]

    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]
    
    possible_answers = []
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)

    answer = majority_voting(possible_answers)
    return answer  
"""
          }

Reflexion = {
    "thought": "For HumanEval, reflexion should focus on debugging the generated code using the task description and prior attempt feedback. The model should revise the implementation when tests or reasoning reveal missing edge cases, syntax problems, or incorrect logic.",
    "name": "Self-Refine",
    "code": """async def forward(self, taskInfo):
    cot_initial_instruction = "Carefully inspect the HumanEval task, reason about the algorithm step by step, and then write a complete Python implementation."

    cot_reflect_instruction = "Given your previous attempt and feedback, identify logical bugs, missing cases, and implementation issues. Revise the Python code so it better matches the task and tests."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Refine Coding Agent')

    critic_instruction = "Review the proposed implementation for syntax issues, missing edge cases, and mismatches with the task. If the code looks correct, output 'True' in 'correct'; otherwise explain the issue in 'feedback'."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Code Critic Agent')
    
    N_max = 5 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    thinking, answer = await cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        cot_inputs.extend([thinking, answer, feedback])

        thinking, answer = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
    return answer
"""
}

LLM_debate = {
    "thought": "For HumanEval, debate can help surface different algorithmic strategies before settling on a final implementation. One agent can propose an approach, another can critique edge cases, and a final agent can merge them into a complete function.",
    "name": "LLM Debate",
    "code": """async def forward(self, taskInfo):
    debate_initial_instruction = "Analyze the task, think step by step about one reasonable implementation strategy, and then provide a complete Python solution."

    debate_instruction = "Given other agents' proposed implementations and reasoning, critique them, incorporate the best ideas, and provide an improved complete Python implementation."
    
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', temperature=0.8, role=role) for role in ['Senior Engineer', 'Test Engineer', 'Python Expert']]

    final_decision_instruction = "Given all the above reasoning and candidate implementations, synthesize the best complete Python implementation. Return code only in the implementation field."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)

    max_round = 2 # Maximum number of debate rounds
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]

    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = await debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = await debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    thinking, answer = await final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
    return answer
"""
}

Take_a_step_back = {"thought": "Let LLM first think about the principles involved in solving this task which could be helpful. By understanding the underlying principles, the model can better reason through the problem and provide a more accurate solution.",
                    "name": "Step-back Abstraction",
                    "code": """async def forward(self, taskInfo):
    principle_instruction = "What algorithmic principles, data structures, or implementation pitfalls are relevant to solving this HumanEval task? First think step by step, then list the useful principles and explain them."

    cot_instruction = "Given the function prompt and the relevant principles, think step by step and then write a complete Python implementation."

    principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Coding Agent')

    thinking, principle = await principle_agent([taskInfo], principle_instruction)

    thinking, answer = await cot_agent([taskInfo, thinking, principle], cot_instruction)
    return answer
"""
                    }

QD = {"thought": "Similar to Quality-Diversity methods, let LLM generate multiple diverse interesting solutions could help. By encouraging the model to explore different reasoning paths, we can increase the chances of finding the best solution.",
      "name": "Quality-Diversity",
      "code": """async def forward(self, taskInfo):
    cot_initial_instruction = "Analyze the HumanEval task carefully, think step by step about one valid algorithm, then provide a complete Python implementation."

    qd_instruction = "Given previous attempts, try a different valid coding strategy or implementation detail, while still solving the same HumanEval task correctly."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Coding Diversity Agent')

    final_decision_instruction = "Given all the above candidate implementations, reason over them carefully and provide the best complete Python implementation."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    possible_answers = []
    thinking, answer = await cot_agent(cot_inputs, cot_initial_instruction, 0)

    possible_answers.extend([thinking, answer])

    for i in range(N_max):
        cot_inputs.extend([thinking, answer])

        thinking, answer = await cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])

    thinking, answer = await final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer
"""
      }

Role_Assignment = {"thought": "Similar to Auto-GPT and expert prompting, we can use dynamic control flow in the design to let the agent decide what expert we should use.",
                   "name": "Dynamic Assignment of Roles",
                   "code": """async def forward(self, taskInfo):
    cot_instruction = "Think step by step about the HumanEval task and then write a complete Python implementation."
    expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in ['Algorithm Engineer', 'Testing Engineer', 'Python Expert', 'Helpful Assistant']]

    routing_instruction = "Given the HumanEval task, choose the best expert for writing the implementation. Choose from: Algorithm Engineer, Testing Engineer, Python Expert."
    routing_agent = LLMAgentBase(['choice'], 'Routing agent')

    choice = (await routing_agent([taskInfo], routing_instruction))[0]

    if 'algorithm' in choice.content.lower():
        expert_id = 0
    elif 'test' in choice.content.lower():
        expert_id = 1
    elif 'python' in choice.content.lower():
        expert_id = 2
    else:
        expert_id = 3 # Default to helpful assistant

    thinking, answer = await expert_agents[expert_id]([taskInfo], cot_instruction)
    return answer
"""
                   }

system_prompt = """You are a helpful assistant. Make sure to return a WELL-FORMED JSON object focused on HumanEval code generation."""

base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve code-generation tasks. Your aim is to design an optimal agent that performs well on HumanEval-style programming problems, where the model must write correct Python implementations from function signatures, docstrings, and tests.

## An example HumanEval task:

**Prompt**:
```python
def truncate_number(number: float) -> float:
    '''Return the decimal part of a positive floating point number.'''
```

**Canonical solution**: `return number % 1.0`

# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
# client = openai.AsyncOpenAI()

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def get_json_response_from_gpt(msg, model, system_message):
    \"""
    Function to get JSON response from GPT model.
    \"""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        reasoning_effort="minimal",
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    \"""
    Base class for an LLM agent.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        # ... implementation ...
        output_fields_and_description = {key: f"Your {key}." if key not in ['implementation', 'code'] else f"Your {key}. Return ONLY a complete Python implementation, with no Markdown fences unless explicitly requested." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt 

    async def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        \"""
        Queries the LLM with provided input information and instruction.
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = await get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    async def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return await self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    async def forward(self, taskInfo) -> Union[Info, str]:
        \"""
        Placeholder method for processing task information.
        
        Args:
        - taskInfo (Info): Task information.
        
        Returns:
        - Answer (Union[Info, str]): Your FINAL Answer. Return either a namedtuple Info or a string containing the final Python implementation.
        \"""
        pass

```

# Discovered architecture archive

Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the pass rate on a validation HumanEval set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:

The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try for code generation, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture.
Finally, the last key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture.
Also, it could be helpful to set the LLM’s role and temperature to further control the LLM’s response. Note that the LLMAgentBase() will automatically parse the output and return a list of “Infos”. You can get the content by Infos.content.
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

## WRONG Implementation examples:

Here are some mistakes you may make:

1. This is WRONG: ```
feedback, correct = await critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = await verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)

```
It is wrong to use "Info('feedback', 'Critic Agent', thinking, 0)". The returned "feedback" from LLMAgentBase is already Info.

2. This is WRONG: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = await verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'

```

First, the len(feedback_info) will not work.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.
Lastly, again, DO NOT CREATE Info object by yourself.

3. This is WRONG: ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
outputs = await agent([taskInfo], independent_reasoning_instruction.format(role=role))
all_thinking.append(outputs[0].content)
all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers

aggregated_thinking = '\n'.join(all_thinking)
aggregated_answers = '\n'.join(all_answers)

```
You SHOULD NOT extract the content from the Info object by yourself. You should use the Info object directly. If you want to aggregate the content, you should just put those Info objects into a list and then use the list as input to the next LLM agent.

4. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = await reasoning_agent([taskInfo] + ..., reasoning_instruction)
    
# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)

```

You should not extract the final answer by yourself. You SHOULD directly return the answer Info. Also, you should always return the best answer you can get.
CORRECT example: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = await reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer

```

# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize "fitness" by proposing interestingly new agents for HumanEval-style code generation. 
Observe the discovered architectures carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative to think about the next interesting architecture to try. You are encouraged to draw inspiration from related LLM agent papers or academic papers from other research areas.
Using the knowledge learned from the archive and the inspiration from academic literature to give the next interesting architecture.
THINK OUTSIDE THE BOX.
"""

Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture for HumanEval and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that better improves code generation accuracy, test coverage, or debugging. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt. Focus on code correctness, syntax, edge cases, and test behavior.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness on HumanEval. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements for HumanEval code generation.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code. The code should output a complete Python solution for a HumanEval task.
"""

Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""


def get_init_archive():
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back, QD, Role_Assignment]
    # return [COT]

def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))

    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2
