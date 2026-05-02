import ast
import json
import os
import openai
import random
import sys
import io
import asyncio
from contextlib import redirect_stdout, redirect_stderr
from CoLLMLP import CoLLMLP
from utils import *

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

PART = int(sys.argv[1])
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]

ACTIVATION = "listwise"
TYPE = "code_completion"
# ROLES = ["Assistant", "Mathematician", "Mathematician", "Assistant"]
DIR_NAME = sys.argv[4]
ROLES = ast.literal_eval(sys.argv[5])
JUDGES = ast.literal_eval(sys.argv[6])
MAX_TASKS = int(sys.argv[7]) if len(sys.argv) > 7 else None
QUIET = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
BATCH_SIZE = int(sys.argv[9]) if len(sys.argv) > 9 else 10
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)

SUBSET = 50

def set_rd_seed(seed):
    random.seed(seed)

async def process_single_problem(task_id, que, entry_point):
    """Run one HumanEval problem in a worker thread (async wrapper)."""
    llmlp = CoLLMLP(MODEL, len(ROLES), ROLES, len(JUDGES), JUDGES, 3, ACTIVATION, TYPE, MODEL)

    def _run_problem():
        llmlp.zero_grad()
        if QUIET:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                res, resp_cnt, completions, prompt_tokens, completion_tokens, tests = llmlp.forward(que, entry_point)
                imp_score = llmlp.backward(res, que, entry_point)
        else:
            res, resp_cnt, completions, prompt_tokens, completion_tokens, tests = llmlp.forward(que, entry_point)
            imp_score = llmlp.backward(res, que, entry_point)

        return {
            "task_id": task_id,
            "completion": completions,
            "final_completion": res,
            "resp_cnt": resp_cnt,
            "importance": imp_score,
            "tests": tests,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    return await asyncio.to_thread(_run_problem)


async def main_async():
    set_rd_seed(0)
    assert len(ROLES) > 0
    assert len(JUDGES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)
    qa_pairs = get_human_eval_qa_pairs()

    # Only process this partition's tasks; optional cap supports quick sample runs.
    part_tasks = []
    for task_id, que, entry_point in qa_pairs:
        qid = int(task_id.split("/")[-1])
        if PART * SUBSET <= qid < (PART + 1) * SUBSET:
            part_tasks.append((task_id, que, entry_point))
    if MAX_TASKS is not None:
        part_tasks = part_tasks[:MAX_TASKS]

    total_tasks = len(part_tasks)
    if total_tasks == 0:
        print(f"No tasks found for part={PART} with max_tasks={MAX_TASKS}.")
        return

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.json', 'w') as f:
        f.write("")
    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.tests', 'w') as f:
        f.write("")

    results, resp_cnts, importances = [], 0, []
    total_prompt_tokens, total_completion_tokens = 0, 0

    progress_bar = tqdm(total=total_tasks, desc=f"Part {PART}", leave=True) if tqdm is not None else None

    for batch_idx in range(0, total_tasks, BATCH_SIZE):
        batch = part_tasks[batch_idx:batch_idx + BATCH_SIZE]
        tasks = [
            process_single_problem(task_id, que, entry_point)
            for task_id, que, entry_point in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if progress_bar is not None:
                progress_bar.update(1)

            if isinstance(result, Exception):
                if not QUIET:
                    print(f"Error processing problem: {result}")
                continue

            results.append({"task_id": result["task_id"], "completion": result["final_completion"]})
            resp_cnts += result["resp_cnt"]
            importances.append(result["importance"])
            total_prompt_tokens += result["prompt_tokens"]
            total_completion_tokens += result["completion_tokens"]

            with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.json', 'a') as f:
                f.write(json.dumps(result["completion"]) + '\n')
            with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.tests', 'a') as f:
                f.write(json.dumps(result["tests"]) + '\n')

    if progress_bar is not None:
        progress_bar.close()

    if not QUIET:
        print(results)
        print(resp_cnts)
        print(importances)
        print(total_prompt_tokens, total_completion_tokens)
    else:
        print(f"Completed part={PART}: {len(results)}/{total_tasks} tasks")

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.txt', 'w') as f:
        f.write(str(resp_cnts) + " " + str(resp_cnts/total_tasks) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos)/total_tasks for pos in zip(*importances)]) + '\n')
        f.write(str(total_prompt_tokens) + " " + str(total_completion_tokens) + '\n')
    
    write_jsonl(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.jsonl', results)

if __name__ == "__main__":
    asyncio.run(main_async())
