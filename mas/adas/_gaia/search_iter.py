import argparse
import copy
import json
import os
import random
import types  # <--- NEW IMPORT REQUIRED
from collections import namedtuple
import asyncio
import backoff
import numpy as np
import openai
from tqdm import tqdm

from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator

# Update with your actual module name if different
from gaia_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from openai import AsyncOpenAI

client = AsyncOpenAI()

from utils import score_gaia, get_all_examples, bootstrap_confidence_interval, random_id

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
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
    assert json_dict is not None
    return json_dict

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def get_json_response_from_gpt_reflect(msg_list, model, temperature=0.8):
    response = await client.chat.completions.create(
        model=model,
        messages=msg_list,
        reasoning_effort="minimal", 
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict

class LLMAgentBase():
    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-5-mini', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        output_fields_and_description = {
            key: f"Your {key}." if 'answer' not in key else f"""Your {key}. If you are asked for a number, dont use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, dont use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""
            for key in self.output_fields
        }
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
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    async def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = await get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            # print(f"[DEBUG] Agent Response: {response_json}")
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(f"[DEBUG] Agent Query Failed: {e}")
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            response_json = {} # Fallback
            for key in self.output_fields:
                response_json[key] = ''
        
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    async def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return await self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentSystem():
    def __init__(self) -> None:
        pass

async def evaluate_forward_fn(args, forward_str):
    # Dynamically define forward()
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    
    # --- CRITICAL FIX: INSTANCE PATCHING ---
    # We create the instance locally and bind the function ONLY to this instance.
    # This prevents parallel agents from overwriting the global class.
    agentSystem = AgentSystem()
    agentSystem.forward = types.MethodType(func, agentSystem)
    # ---------------------------------------

    # Set seed 0 for reproducible sets
    examples = get_all_examples()
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
        print(f"[DEBUG] Using validation set: {len(examples)} examples")
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat
        print(f"[DEBUG] Using test set: {len(examples)} examples")

    questions = [example['Question'] for example in examples]
    answers = [example['answer'] for example in examples]

    task_queue = [Info('task', 'User', q, -1) for q in questions]

    @llm_parallel_search_decorator
    async def process_task_wrapper(taskInfo, **kwargs):
        res = await agentSystem.forward(taskInfo)
        print(f"[DEBUG] Task processed. Result: {res}")
        return res
    
    acc_list = []
    
    # Use asyncio.gather for concurrency
    sem = asyncio.Semaphore(args.max_workers if args.multiprocessing else 1)
    async def sem_task(task):
        async with sem:
            return await process_task_wrapper(
                task, 
                question=task.content, 
                task_type="qa" 
            )
            
    # return_exceptions=True ensures one failure doesn't kill the batch
    results = await asyncio.gather(*(sem_task(task) for task in task_queue), return_exceptions=True)

    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, Exception):
                raise res
            
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            
            correct_answer = answers[q_idx]
            correct = score_gaia(correct_answer, extracted_answer)
            print(f"[DEBUG] Q{q_idx}: pred={extracted_answer}, gold={correct_answer}, correct={correct}")
        except Exception as e:
            print(f"[DEBUG] Exception in scoring Q{q_idx}: {e}")
            acc_list.append(0)
            continue

        acc_list.append(1 if correct else 0)
    
    print(f"[DEBUG] Final accuracy: {bootstrap_confidence_interval(acc_list)}")
    return acc_list

async def search(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    
    # Load existing archive or init new
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        start = archive[-1]['generation'] if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int) else 0
    else:
        archive = get_init_archive()
        start = 0

    # Evaluate Initial Archive
    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = await evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print(f"Error evaluating initial archive {solution['name']}: {e}")
            continue

        solution['fitness'] = bootstrap_confidence_interval(acc_list)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    # Search Loop
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        try:
            # Initial Proposal
            next_solution = await get_json_response_from_gpt_reflect(msg_list, args.model)

            # Reflexion Loop
            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = await get_json_response_from_gpt_reflect(msg_list, args.model)
            
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = await get_json_response_from_gpt_reflect(msg_list, args.model)
            
        except Exception as e:
            print(f"Error generating solution: {e}")
            n -= 1
            continue

        # Evaluate Proposed Solution (with retry/debug)
        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = await evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print(f"Error during evaluation: {e}")
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong..."})
                try:
                    next_solution = await get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as inner_e:
                    print(f"Error during debug generation: {inner_e}")
                    break # Break inner loop to retry outer loop
                continue
        
        if not acc_list:
            print("Failed to get valid accuracy after retries.")
            n -= 1
            continue

        next_solution['fitness'] = bootstrap_confidence_interval(acc_list)
        next_solution['generation'] = n + 1
        
        # Cleanup response
        next_solution.pop('debug_thought', None)
        next_solution.pop('reflection', None)
        
        archive.append(next_solution)

        # Save results
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

async def evaluate(args):
    """
    Evaluates all agents in the archive in parallel.
    """
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    # eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    eval_file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json").replace(".json", "") + "_evaluate.json"
    
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    # Determine which agents need evaluation
    # We skip agents that are already present in eval_archive
    # (Simple check based on length, assumes append-only)
    agents_to_evaluate = []
    start_idx = len(eval_archive)
    
    if start_idx < len(archive):
        agents_to_evaluate = [(i, archive[i]) for i in range(start_idx, len(archive))]

    print(f"Starting parallel evaluation for {len(agents_to_evaluate)} agents...")

    # Wrapper function to evaluate a single agent and return the full object
    async def eval_single_agent(idx, sol):
        print(f"Evaluating Agent {idx} (Gen {sol.get('generation')})...")
        try:
            acc_list = await evaluate_forward_fn(args, sol["code"])
            sol['test_fitness'] = bootstrap_confidence_interval(acc_list)
            return sol
        except Exception as e:
            print(f"Failed to evaluate Agent {idx}: {e}")
            return None

    # Run all evaluations concurrently
    tasks = [eval_single_agent(idx, sol) for idx, sol in agents_to_evaluate]
    results = await asyncio.gather(*tasks)

    # Filter successful results and save
    new_results = [r for r in results if r is not None]
    eval_archive.extend(new_results)

    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    with open(eval_file_path, 'w') as json_file:
        json.dump(eval_archive, json_file, indent=4)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size', type=int, default=20)
    parser.add_argument('--test_size', type=int, default=83)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='judge_iter_results/')
    parser.add_argument('--expr_name', type=str, default="gpt5_results")
    parser.add_argument('--n_generation', type=int, default=1)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-5-mini', choices=['gpt-5-mini', 'gpt-5-nano'])

    args = parser.parse_args()

    async def main():
        global SEARCHING_MODE
        
        # Phase 1: Search (Discovery)
        SEARCHING_MODE = True
        print("Starting SEARCH Phase...")
        await search(args)
        
        # Phase 2: Evaluate (Testing)
        SEARCHING_MODE = False
        print("Starting EVALUATE Phase...")
        await evaluate(args)

    asyncio.run(main())