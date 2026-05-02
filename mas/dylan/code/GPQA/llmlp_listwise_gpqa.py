import ast
import asyncio
import json
import os
import random
import sys

from LLMLP import LLMLP
from utils import get_gpqa_qa_pairs


QUERY_DIR = sys.argv[1]
MIN_FILE = int(sys.argv[2])
MAX_FILE = int(sys.argv[3])
EXP_NAME = sys.argv[4]
MODEL = sys.argv[5]
DIR_NAME = sys.argv[6]
ROLES = ast.literal_eval(sys.argv[7])
ASYNC_BATCH_SIZE = int(sys.argv[8]) if len(sys.argv) > 8 else 20
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)

ACTIVATION = "listwise"
TYPE = "single_choice"


def set_rd_seed(seed):
    random.seed(seed)


def normalize_choice(answer):
    if answer is None:
        return ""
    return str(answer).strip().upper()


async def process_single_problem(que, ans, roles, model, activation, qtype):
    llmlp = LLMLP(model, len(roles), roles, 3, activation, qtype, model)

    def _run_problem():
        llmlp.zero_grad()
        res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(que)
        imp_score = llmlp.backward(res)
        return {
            "completion": completions,
            "acc": normalize_choice(ans) == normalize_choice(res),
            "resp_cnt": resp_cnt,
            "importance": imp_score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    return await asyncio.to_thread(_run_problem)


async def main_async():
    set_rd_seed(0)
    assert len(ROLES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    qa_pairs = get_gpqa_qa_pairs(QUERY_DIR, MIN_FILE, MAX_FILE)

    with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.json', 'w', encoding='utf-8') as f:
        f.write("")

    accs, resp_cnts, importances = [], 0, []
    completion_list = []
    total_prompt_tokens, total_completion_tokens = 0, 0

    print(f"Processing {len(qa_pairs)} GPQA problems in async batches...")

    for batch_idx in range(0, len(qa_pairs), ASYNC_BATCH_SIZE):
        batch = qa_pairs[batch_idx:batch_idx + ASYNC_BATCH_SIZE]
        tasks = [
            process_single_problem(que, ans, ROLES, MODEL, ACTIVATION, TYPE)
            for que, ans in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                # print(f"Error processing problem: {result}")
                continue

            completion_list.append(result['completion'])
            accs.append(result['acc'])
            resp_cnts += result['resp_cnt']
            importances.append(result['importance'])
            total_prompt_tokens += result['prompt_tokens']
            total_completion_tokens += result['completion_tokens']

            with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(result['completion']) + '\n')

        processed = len(accs)
        if processed > 0:
            print(f"Processed {processed}/{len(qa_pairs)} | current acc={sum(accs)/processed:.4f}")

    # print(accs)
    # print(resp_cnts)
    # print(importances)

    with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.txt', 'w', encoding='utf-8') as f:
        f.write(str(accs) + ' ' + str(sum(accs) / len(qa_pairs) if len(qa_pairs) > 0 else 0.0) + '\n')
        f.write(str(resp_cnts) + " " + str(resp_cnts / len(qa_pairs) if len(qa_pairs) > 0 else 0.0) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos) / len(qa_pairs) for pos in zip(*importances)] if len(qa_pairs) > 0 else []) + '\n')
        f.write(str(total_prompt_tokens) + ' ' + str(total_completion_tokens) + '\n')


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()