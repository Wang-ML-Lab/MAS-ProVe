import copy
import functools
import inspect
import json
import os
import re
import types
from collections import namedtuple
from typing import Tuple, List

import numpy as np
import openai
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

import common
from common import ANSWER_PATTERN, shorten_context, merge_context
from common import HTML_JINJA, get_init_archive_local, get_prompt_local, SingleEvalResult, get_reflexion_after_eval_local
from common import get_json_response_from_gpt_local, get_json_response_from_gpt_reflect_local, _pack_message, get_json_response_from_gpt_local_parallel
from prompts.swe.patch_oracle import AGENTLESS_REPAIR
from utils import extract_xml
from score import DataScorer
from utils import random_id, bootstrap_confidence_interval
from sampler import get_model
import asyncio

client = openai.OpenAI()

Info = namedtuple('Info', ['name', 'author', 'content', 'prompt', 'sub_tasks', 'agents', 'iteration_idx'])

json_next_step_prompt = """{prev_info}Given the above, answer the following question: {instruction}

If the question is too complicated or information is missing, you still need to give your best answer but add \
(1) an additional mark [TOO_HARD] in the next line of your final answer \
(2) information request or decomposition suggestion in the next line of the [TOO_HARD] mark, in the "answer" entry (for example, 300
[TOO_HARD]
Suggestion:...) and justify why you think so in the "thinking" entry"""

xml_next_step_prompt = """{prev_info}Given the above, answer the following question: {instruction}

If the question is too complicated or information is missing, you still need to give your best guess but add \
(1) an additional mark [TOO_HARD] in the next line of your final answer \
(2) information request or decomposition suggestion in the next line of the [TOO_HARD] mark, in the "answer" entry. \
In the "thinking", justify why you think so. Following the format below:

"answer" entry: [Your best guess, e.g., 300]\n[TOO_HARD]\nSuggestion: [your suggestion]
"thinking" entry:  [why you thinking is is too complicated or missing information. How to you arrive your best guess regardless]

Otherwise, give your answer and thinking normally.

"answer" entry: [your answer]
"thinking" entry: [How do you arrive your answer]

IMPORTANT: You need to give your best guess in both cases. Do not give [TOO_HARD] directly but always give your best guess first

"""


def role2desc(role):
    return f"You are a {role}."


class LLMAgentBase:
    def __init__(self,
                 output_fields: list,
                 agent_name: str,
                 role='helpful assistant',
                 model=None,
                 temperature=None):
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature
        # give each instance a unique id
        self.id = random_id()

    def extract_pattern(self, prompt):
        # pattern = r"\s*(.*?)\s*\n\nRelated original question"
        pattern = r"Given the above, answer the following question: \s*(.*?)\s*\n\n"

        sub_question = prompt[-1]['content']
        match = re.search(pattern, sub_question, re.DOTALL)
        extracted_question = match.group(1)

        return extracted_question

    def generate_prompt(self, input_infos, extra_info, instruction, is_sub_task=False) -> Tuple[str, str]:

        output_description = extra_info["output_description"]
        format_inst = extra_info["FORMAT_INST"]

        format_choice = extra_info["format_choice"]

        if format_choice == 'json':
            output_fields_and_description = {key: f"Your {key}." if 'answer' not in key else f"Your {key}. {output_description}" for key in self.output_fields}
        elif format_choice == 'xml':
            output_fields_and_description = '\n'.join(
                [
                    f"<{key}> [Your {key}.] </{key}>" if 'answer' not in key else f"<{key}> [Your {key}. {output_description}] </{key}>\n"
                    for key in self.output_fields
                ])
        else:
            raise NotImplementedError

        system_prompt = role2desc(self.role) + "\n\n" + format_inst.format(request_keys=output_fields_and_description)
        
        # construct input infos text
        input_infos_text = ''
        prev_extracted_question = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, prompt, _, _, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                if is_sub_task:
                    input_infos_text += f'Related original question:\n\n{content}. \n\nRelated sub-task questions and answers:\n\n'
                else:
                    input_infos_text += f'{content}\n\n'
            elif iteration_idx != -1:
                if is_sub_task and prompt is not None:
                    extracted_question = self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question
                    else:
                        input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'

                else:
                    input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                if is_sub_task and prompt is not None:
                    extracted_question = self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question  # we do not want to duplicate the prompt
                    else:
                        input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        if is_sub_task:
            if format_choice == 'json':
                prompt = json_next_step_prompt.format(prev_info=input_infos_text, instruction=instruction)  # instruction (sub-task in above)
            elif format_choice == 'xml':
                prompt = xml_next_step_prompt.format(prev_info=input_infos_text, instruction=instruction)
            else:
                raise NotImplementedError
        else:
            prompt = input_infos_text + instruction
        return system_prompt, prompt

    async def query(self, input_infos: list, extra_info: dict, instruction, iteration_idx=-1, is_sub_task=False) -> List:

        system_prompt, prompt = self.generate_prompt(input_infos, extra_info, instruction, is_sub_task=is_sub_task)

        prompt = [
            _pack_message(content=system_prompt, role="system"),
            _pack_message(content=prompt, role="user")]
        
        dataset = extra_info.get('dataset', '')
        task_type = "math" if "aime" in dataset else "qa" if "gaia" in dataset else "math"
        
        # 2. Extract Question
        question = extra_info.get('questions', '')
        
        # print("\nThis is input infos", input_infos)
        # print("\n This is extra info:", extra_info)
        # response_json = await get_json_response_from_gpt_local(prompt, self.model, self.output_fields, self.temperature, extra_info,trajectory=trajectory_context, task_type=task_type, question=question)
        
        if isinstance(extra_info.get('n', -1), int):
            trajectory_context = []
            for info in input_infos:
                # Only include if author is not 'User'
                if hasattr(info, 'author') and str(info.author).lower() != 'user' and hasattr(info, 'content'):
                    trajectory_context.append(f"[{info.author}]: {info.content}")
            response_json = await get_json_response_from_gpt_local_parallel(prompt, self.model, self.output_fields, self.temperature, extra_info,trajectory=trajectory_context, task_type=task_type, question=question)
        else:
            response_json = await get_json_response_from_gpt_local(prompt, self.model, self.output_fields, self.temperature, extra_info)
        # print("\n This is the response from agent:", self.__repr__(), "content:", response_json)
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, prompt, None, None, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    async def __call__(self, input_infos: list, extra_info, instruction, iteration_idx=-1, is_sub_task=False):
        return await self.query(input_infos, extra_info, instruction, iteration_idx=iteration_idx, is_sub_task=is_sub_task)


class AgentSystem:

    def make_final_answer(self, thinking, answer, sub_tasks=None, agents=None):

        name = thinking.name
        author = thinking.author
        prompt = thinking.prompt
        iteration_idx = thinking.iteration_idx

        if type(answer) == str:
            answer_content = answer
        else:
            answer_content = answer.content

        if agents is None:  # this means sub_task is None, according to the propose prompt
            sub_tasks, agents = agents, sub_tasks

        if sub_tasks is None and agents is None:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', prompt, None, None, iteration_idx)
        elif agents is not None:  # when remove decomposition, we still have agent output logged
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', prompt, None, '\n'.join(agents), iteration_idx)
        else:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', prompt, '\n'.join(sub_tasks), '\n'.join(agents), iteration_idx)
        return final_answer


async def evaluate_forward_fn(extra_info, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py

    print('forward_str: ', forward_str)

    # if you want debug, remove the section so that you can see the detailed error line
    namespace = {}
    global_env = dict(extra_info)
    global_env.update({
        "AgentSystem": AgentSystem,
        "LLMAgentBase": LLMAgentBase,
        "Info": Info,
    })
    print("This is a sub-agent call\n")
    exec(forward_str, global_env, namespace)  # This defines a `forward` function here
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")

    agent_system = AgentSystem()
    # Assign the function to this instance only without affecting other instances across threads
    agent_system.forward = types.MethodType(func, agent_system)

    # global_max_workers = extra_info["max_workers"]
    task_queue = extra_info["task_queue"]
    answers = extra_info["answers"]
    technique = extra_info["dataset"].split('/')[0]
    data_scorer = DataScorer(extra_info["dataset"], technique, extra_info["verifier_model"])
    agent_system.node_model = extra_info["node_model"]
    agent_system.cot_instruction = extra_info["cot_instruction"]
    agent_system.max_sc = extra_info["max_sc"]
    agent_system.max_round = extra_info["max_round"]
    agent_system.debate_role = extra_info["debate_role"]
    agent_system.dataset = extra_info["dataset"]
    agent_system.example_id = extra_info["example_id"]
    agent_system.instance_id = extra_info["instance_id"]

    # tasks = [agent_system.forward(item) for item in task_queue]
    # results = await tqdm_asyncio.gather(*tasks, desc="Evaluating forward function", total=len(tasks))
    results = []
    for item in tqdm(task_queue, desc="Evaluating forward function", total=len(task_queue)):
        res = await agent_system.forward(item, extra_info)
        results.append(res)

    prompt_messages = [res.prompt for q_idx, res in enumerate(results)]
    response_texts = [str(res.content) for q_idx, res in enumerate(results)]
    if not extra_info["no_decompose"]:
        sub_tasks = [res.sub_tasks for q_idx, res in enumerate(results)]
        sub_tasks_text = sub_tasks[0]  # only one sample
    else:
        sub_tasks = None
        sub_tasks_text = None

    agents = [res.agents for q_idx, res in enumerate(results)]

    # print('response_texts: ', response_texts[0])
    # print('gold answers: ', answers[0])
    # print('length: ', len(response_texts), len(answers))

    example_id = extra_info["example_id"]
    n = extra_info["n"]
    questions = extra_info["questions"]
    answers = extra_info["answers"]
    use_oracle_verifier = extra_info["use_oracle_verifier"]
    judge_path = extra_info["judge_path"]
    response_path = extra_info["response_path"]
    response_dict = extra_info["response_dict"]
    instance_id = extra_info["instance_id"]
    code_snippet = extra_info["code_snippet"]

    result_list = [
        await data_scorer.score(
            example_id,
            n,
            prompt_messages[response_text_id],
            questions[response_text_id],
            response_text,
            answers[response_text_id],
            sub_tasks_text,
            use_oracle_verifier,
            judge_path,
            response_path,
            response_dict,
            instance_id,
            code_snippet)
        for response_text_id, response_text in enumerate(response_texts)
    ]

    acc_oracle_verifier_list = [x[0] for x in result_list]
    acc_model_verifier_list = [x[1] for x in result_list]
    result_list = [x[2] for x in result_list]
    results = common.aggregate_results(result_list)

    print(f"acc_oracle_verifier_list:", acc_oracle_verifier_list)
    print(f"acc_model_verifier_list:", acc_model_verifier_list)

    return acc_oracle_verifier_list, acc_model_verifier_list, results, sub_tasks, agents, response_texts


async def search(extra_info, task_queue, meta_model, blocks, verifier_model, n_generation,
                 save_dir, expr_name, option, dataset, defer_verifier, debug_max):
    questions = extra_info["questions"]
    node_model = extra_info["node_model"]

    print(f"a new search start")

    print(f"problem length: {len(questions)}")

    task_queue = [Info(field_name, author, content, prompt, sub_tasks, agents, iteration_idx) for
                  field_name, author, content, prompt, sub_tasks, agents, iteration_idx in task_queue]

    # extra_info["global_max_workers", max_workers)
    # extra_info["global_task_queue", task_queue)
    # extra_info["max_workers"] = max_workers
    extra_info["task_queue"] = task_queue

    next_solution_path = os.path.join(save_dir, f"{expr_name}_{option}_next_solution.json")
    msg_path = os.path.join(save_dir, f"{expr_name}_{option}_msg.json")
    mem_path = os.path.join(save_dir, f"{expr_name}_{option}_mem.json")
    file_path = os.path.join(save_dir, f"{expr_name}_{option}_archive.json")
    result_path = f'./async_results/question/meta_agent/{dataset}/{meta_model}_{node_model}_{verifier_model}.results'
    oracle_acc_result_path = f'./async_results/question/meta_agent/{dataset}/{meta_model}_{node_model}_oracle.results'
    judge_path = os.path.join(save_dir, f"{expr_name}_{option}_judge")
    response_path = os.path.join(save_dir, f"{expr_name}_{option}_response")
    os.makedirs(os.path.dirname(judge_path), exist_ok=True)

    print('file_path: ', file_path)
    print('msg_path: ', msg_path)
    print('result_path: ', result_path)
    print('next_solution_path: ', next_solution_path)
    print('oracle_acc_result_path: ', oracle_acc_result_path)
    print('judge_path: ', judge_path)
    print('response_path: ', response_path)
    print('mem_path: ', mem_path)

    extra_info["judge_path"] = judge_path
    extra_info["response_path"] = response_path

    if os.path.exists(mem_path):
        with open(mem_path, 'r') as json_file:
            memory = json.load(json_file)
    else:
        memory = []

    if os.path.exists(response_path):
        with open(response_path, 'r') as json_file:
            response_dict = json.load(json_file)
        extra_info["response_dict"] = response_dict

    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]["generation"], int):
            start = archive[-1]["generation"]
        else:
            start = 0
    else:
        archive = get_init_archive_local(blocks, extra_info)  # TODO: make this with argument
        start = 0

    cur_archive = copy.deepcopy(archive)  # do not change the solution inside, need deepcopy

    use_oracle_verifier = extra_info["use_oracle_verifier"]
    example_id = extra_info["example_id"]

    global_ns = []
    # Temporarily disable the initial archive evaluation
    for solution_i, solution in enumerate(cur_archive):

        if 'fitness' in solution:
            continue

        solution["generation"] = "initial"
        print(f'============Initial Archive: {solution["name"]}=================')

        if solution["name"] in global_ns:  # TODO: separate it
            extra_info["n"] = f'{solution["name"]}_{solution_i}'
        else:
            extra_info["n"] = solution["name"]

        global_n = extra_info["n"]
        global_ns.append(global_n)

        # print(solution["code"])
        try:
            acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_response = await evaluate_forward_fn(extra_info, solution["code"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            # print(solution['code'])
            continue

        # TODO: can we somehow also log acc_oracle_verifier_list so that we can know how accurate acc_model_verifier_list is?
        if use_oracle_verifier:
            acc_list = acc_oracle_verifier_list
        else:
            acc_list = acc_model_verifier_list

        if defer_verifier:
            fitness_str = bootstrap_confidence_interval([0.0])
            solution["acc"] = np.mean([0.0])

        else:
            fitness_str = bootstrap_confidence_interval(acc_list)
            solution["acc"] = np.mean(acc_list)

        solution["fitness"] = fitness_str
        solution["total_cost"] = extra_info["COST_TOTAL"]

        print(f"acc_list:", acc_list)
        print(f"mean acc_list:", np.mean(acc_list))
        print(f"bootstrap_confidence_interval: {fitness_str}")

        if 'swe_bench' in dataset:
            extracted_answer = final_response[0].split('\n\nAnswer:', 1)[-1].strip()
            if '<patch>' in extracted_answer:
                extracted_answer = extract_xml(extracted_answer, 'patch').strip()
        else:
            extracted_answer = re.search(ANSWER_PATTERN, final_response[0]).group(1)

        if '[TOO_HARD]' in extracted_answer:  # we cannot add [TOO_HARD] in memory
            extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]
        memory.append({extracted_answer: fitness_str})
        print(f'save json to {mem_path}')
        with open(mem_path, 'w') as json_file:
            json.dump(memory, json_file, indent=4)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f'save json to {file_path}')
        with open(file_path, 'w') as json_file:
            json.dump(cur_archive, json_file, indent=4)

        # report_filename = os.path.join(save_dir, f'{expr_name}_{solution["name"]}_{option}_debug.html')
        # print(f"Writing report to {report_filename}")
        # with open(report_filename, "w") as fh:
        #     fh.write(common.make_report(results))
        metrics = results.metrics | {"score": results.score}
        print('metrics: ', metrics)
        print(f"COST_TOTAL:", extra_info["COST_TOTAL"])

        # with open(oracle_acc_result_path, "a+") as fh:
        #     fh.write(
        #         f'experiment {example_id}: 1 (initial {solution["name"]}): acc_oracle_verifier_list: {acc_oracle_verifier_list} '
        #         f'acc_model_verifier_list: {acc_model_verifier_list}\n')

        # if not defer_verifier:
        #     if np.mean(acc_list) == 1:
        #         if use_oracle_verifier:
        #             with open(result_path, "a+") as fh:
        #                 fh.write(f'experiment {example_id}: 1 (initial {solution["name"]})\n')

        #         else:
        #             # check with the real answer to decide whether to mark as correct
        #             if np.mean(acc_oracle_verifier_list) == 1:  #
        #                 with open(result_path, "a+") as fh:
        #                     fh.write(f'experiment {example_id}: 1 (initial {solution["name"]})\n')

        #         # even the judge is incorrect, we still stop because have to listen to the judge
        #         n_generation = 0  # no need
        #         start = 0  # no need
        #         print(f'write to {result_path}. break')
        #         break

        #     if acc_oracle_verifier_list[0] == 1:
        #         exit()  # debug
    # exit()

    task_queue = extra_info["task_queue"]
    format_choice = extra_info["format_choice"]

    # print(f"Task queue: {task_queue}")

    for n in range(start, n_generation):
        print(f"============Generation {n + 1}=================")
        extra_info["n"] = n

        # if n == 0:  # initial propose
        #     system_prompt, prompt = get_prompt_local(cur_archive, extra_info, option=args.option, task_queue=task_queue)
        #     msg_list = [
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": prompt},
        #     ]
        #
        #     next_solution = get_json_response_from_gpt_reflect_local(copy.deepcopy(msg_list), meta_model, extra_info)

        if os.path.exists(msg_path):
            print(f'load msg_list from {msg_path}')
            with open(msg_path, 'r') as json_file:
                msg_list = json.load(json_file)  # use the saved msg_list
            # print("Message :", msg_list)
        if os.path.exists(next_solution_path):
            print(f'load next_solution_path from {next_solution_path}')
            with open(next_solution_path, 'r') as json_file:
                next_solution = json.load(json_file)  # use the saved msg_list

        else:
            # if no next solutionm, you have to do it again
            system_prompt, prompt = get_prompt_local(cur_archive, format_choice, extra_info["no_decompose"], extra_info["no_meta_reward"],
                                                     option=option, task_queue=task_queue)

            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": "Initial Round (Round 0):\n\n" + prompt + '\n\nIMPORTANT: You must follow all the requirements in the Initial Round (Round 0) '
                                                                      '(e.g., what is wrong and correct in code implementation; '
                                                                      'You need to ACTUALLY IMPLEMENT the structure for self-consistency, '
                                                                      'LLM Debate and Reflexion, by writing the for-loop, if you choose to use them).'},
            ]

            next_solution = await get_json_response_from_gpt_reflect_local(copy.deepcopy(msg_list), meta_model, extra_info)
            # print('next_solution: ', next_solution['code'])
        acc_list = []
        agents = None
        acc_oracle_verifier_list = []
        acc_model_verifier_list = []
        results = None
        sub_tasks = None
        final_response = None
        try:
            acc_oracle_verifier_list, acc_model_verifier_list, results, agents, sub_tasks, final_response = await evaluate_forward_fn(extra_info, next_solution["code"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            # print(solution['code'])
            continue
        if use_oracle_verifier:
            acc_list = acc_oracle_verifier_list
        else:
            acc_list = acc_model_verifier_list

        
        if defer_verifier:
            fitness_str = bootstrap_confidence_interval([0.0])
            next_solution["acc"] = [0.0]

        else:
            fitness_str = bootstrap_confidence_interval(acc_list)
            next_solution["acc"] = np.mean(acc_list)

        # Only have these after excusion
        next_solution["fitness"] = fitness_str
        next_solution["generation"] = n + 1
        next_solution["total_cost"] = extra_info["COST_TOTAL"]

        if not (extra_info["no_decompose"] or extra_info["no_meta_reward"]):
            next_solution["sub_tasks"] = sub_tasks
        if not extra_info["no_meta_reward"]:
            next_solution["agents"] = agents

        next_solution["final_response"] = final_response

        if 'swe_bench' in dataset:
            extracted_answer = final_response[0].split('\n\nAnswer:', 1)[-1].strip()
            if '<patch>' in extracted_answer:
                extracted_answer = extract_xml(extracted_answer, 'patch').strip()
        else:
            if final_response is None:
                extracted_answer = '[TOO_HARD]'
            else:
                extracted_answer = re.search(ANSWER_PATTERN, final_response[0]).group(1)

        if '[TOO_HARD]' in extracted_answer:
            extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]
        memory.append({extracted_answer: fitness_str})
        print(f'save json to {mem_path}')
        with open(mem_path, 'w') as json_file:
            json.dump(memory, json_file, indent=4)

        cur_archive.append(next_solution)  # propose

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f'save json to {file_path}')
        with open(file_path, 'w') as json_file:
            json.dump(cur_archive, json_file, indent=4)

        print(f"COST_TOTAL:", extra_info["COST_TOTAL"])

        # with open(oracle_acc_result_path, "a+") as fh:
        #     fh.write(
        #         f'experiment {example_id}: 1 (generation {n}+1): acc_oracle_verifier_list: {acc_oracle_verifier_list} '
        #         f'acc_model_verifier_list: {acc_model_verifier_list}\n')

        # if not defer_verifier:
        #     if np.mean(acc_list) == 1:
        #         if use_oracle_verifier:
        #             with open(result_path, "a+") as fh:
        #                 fh.write(f'experiment {example_id}: 1 (generation {n}+1) \n')
        #         else:
        #             # check with the real answer to decide whether to mark as correct
        #             if np.mean(acc_oracle_verifier_list) == 1:  #
        #                 with open(result_path, "a+") as fh:
        #                     fh.write(f'experiment {example_id}: 1 (generation {n}+1) \n')

        #         print(f'write to {result_path}. break')
        #         break  # good enough

        # not good, need update again %%%%%%%%%%%%%
        Reflexion_after_eval_prompt = get_reflexion_after_eval_local(option, extra_info["format_choice"], extra_info["no_decompose"],
                                                                     extra_info["no_meta_reward"])

        if 'workflow_search' in dataset and 'swe_bench' in dataset:
            code_snippet = extra_info["code_snippet"]
            Reflexion_after_eval_prompt = f'Recall the requirement of original questions: \n\nGiven code_snippet \n\n{code_snippet}; Generate a patch following requirements: {AGENTLESS_REPAIR} \n\n Now please ' + Reflexion_after_eval_prompt + f'\n\nIMPORTANT Note: The above "code" entry is only for the code of your improved architecture and sub-tasks.'
            # For example: {EXAMPLE_META} # Add Example may make the output patch worse

        # recall the xml format
        if format_choice == 'xml':
            Reflexion_after_eval_prompt += "IMPORTANT: 1. Make sure to return in a WELL-FORMED XML object. Wrap the required entries with <(entry_name)> and </(entry_name)>. Reply EXACTLY with the following XML fileds.\n<reflection> [Your reflection] </reflection>\n<thought> [Your thought.] </thought>\n<name> [Your name.] </name>\n<code> [Your code.] </code>\n\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object! However, Do not use XML format inside each entry\n\n2. You must follow all the requirements in the Initial Round (Round 0) (for example, If you choose to use self-consistency, LLM Debate or Relexion, you need to ACTUALLY IMPLEMENET their structure by wrting the for-loop explictly).\n\n3.the <code> corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in <code>: Your code will be part of the entire project (so do not implement any other part), so please implement complete, reliable, reusable code snippets."

        next_solution["memory"] = memory  # TODO: it may output 128K limit
        # print('memory: ',memory) # TODO: Too large, we may not need to print it
        msg_list.append({"role": "assistant", "content": copy.deepcopy(next_solution)})
        # msg_list.append({"role": "user", "content": Reflexion_after_eval_prompt})
        # TODO: We need to handle the multi-turn repeating issues
        msg_list.append({"role": "user",
                         "content": f'Round {n + 1}: The entries (code, thoughts, agents, reflection, etc.) have been updated since last round (Round {n}). Now Using insights from previous rounds, reflect again on the new outputs after round {n}.\n\n' + Reflexion_after_eval_prompt.replace(
                             'round [last_round]', f'round {n}').replace('round [last_last_round]', f'round {n - 1}')})

        _shorten_context = extra_info["shorten_context"]
        _merge_context = extra_info["merge_context"]

        if _shorten_context:
            msg_list_reflect = shorten_context(msg_list)  # the maximum length is limited, we cannot use all
        else:
            msg_list_reflect = msg_list  # if you have enough length

        if _merge_context:  # merge to single turn
            msg_list_reflect = merge_context(msg_list_reflect)

            # TODO: do we want more previous sampeld? we need to be careful about the max limit for qwen

        next_solution = await get_json_response_from_gpt_reflect_local(copy.deepcopy(msg_list_reflect), meta_model,
                                                                       extra_info)  # deep copy to avoid in-place changes
        if next_solution == 'bad_request':
            print('bad_request; break fo now')
            break

        # meta agent results html --------
        # prompt_message = []
        # # TODO: let's look at the reflect msg, to see whether it is correct (04/15)
        # for msg in msg_list_reflect:  # want a better output
        #     message = {'role': msg["role"]}
        #     if msg["role"] == 'assistant':
        #         try:
        #             message["content"] = '\n\n'.join([f'{key}: {item}' for key, item in msg["content"].items()])
        #         except Exception as e:
        #             print("content e: ", e)
        #             message["content"] = msg["content"]
        #     else:
        #         message["content"] = msg["content"]
        #     prompt_message.append(message)

        # response_text = '\n\n'.join([f'{key}: {item}' for key, item in next_solution.items()])

        # html = common.jinja_env.from_string(HTML_JINJA).render(
        #     prompt_messages=prompt_message,
        #     next_message=dict(content=response_text, role="assistant"),
        #     score=0,
        #     correct_answer=0,
        #     extracted_answer=0,
        # )
        # convo = prompt_message + [dict(content=response_text, role="assistant")]
        # results = SingleEvalResult(html=html, score=0, convo=convo)
        # results = common.aggregate_results([results])
        # report_filename = os.path.join(save_dir, f'{expr_name}_{next_solution["name"].strip()}_{option}_generation_{n}_debug.html')
        # print(f"Writing report to {report_filename}")
        # with open(report_filename, "w") as fh:
        #     fh.write(common.make_report(results))
        # meta agent results html -------

        if 'debug_thought' in next_solution:
            del next_solution["debug_thought"]

        with open(msg_path, 'w') as json_file:
            json.dump(msg_list, json_file, indent=4)
        with open(next_solution_path, 'w') as json_file:
            json.dump(next_solution, json_file, indent=4)

        if not defer_verifier:
            if acc_oracle_verifier_list[0] == 1:
                exit()  # debug
