import ast
import asyncio
import json
import math
import os
import random
import sys

from LLMLP import LLMLP
from LLM_Neuron import LLMEdge, LLMNeuron, listwise_ranker_2
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator
from utils import get_gpqa_qa_pairs, most_frequent, parse_single_choice

QUERY_DIR = sys.argv[1]
MIN_FILE = int(sys.argv[2])
MAX_FILE = int(sys.argv[3])
EXP_NAME = sys.argv[4]
MODEL = sys.argv[5]
DIR_NAME = sys.argv[6]
ROLES = ast.literal_eval(sys.argv[7])
ASYNC_BATCH_SIZE = int(sys.argv[8]) if len(sys.argv) > 8 else 20

ACTIVATION = "listwise"
TYPE = "single_choice"
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)


class LLMLP_GPQA(LLMLP):
    """GPQA LLMLP with AIME-style iterative round search via decorated round executors."""

    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=3, activation="listwise", qtype="single_choice", mtype="gpt-5-mini"):
        super().__init__(default_model_name, agents, agent_roles, rounds, activation, qtype, mtype)
        self.cmp_res = lambda x, y: x == y
        self.ans_parser = parse_single_choice

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, parse_single_choice, self.qtype))

        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, parse_single_choice, self.qtype))
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")

    @llm_parallel_search_decorator
    async def _execute_round_0(self, question, **kwargs):
        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)

        activated_indices = []
        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0

        for idx, node_idx in enumerate(loop_indices):
            await asyncio.to_thread(self.nodes[node_idx].activate, question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)

            if idx >= math.floor(2 / 3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    trace = "=== Round 0 (Consensus Reached) ===\n"
                    for a_idx in activated_indices:
                        trace += f"Agent {a_idx}: {self.nodes[a_idx].get_reply()}...\n"

                    return {
                        "response": trace,
                        "consensus_reached": True,
                        "consensus_reply": reply,
                        "activated_indices": activated_indices,
                        "resp_cnt": resp_cnt,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    }

        trace = "=== Round 0 ===\n"
        for a_idx in activated_indices:
            trace += f"Agent {a_idx}: {self.nodes[a_idx].get_reply()}...\n"

        return {
            "response": trace,
            "consensus_reached": False,
            "activated_indices": activated_indices,
            "resp_cnt": resp_cnt,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

    @llm_parallel_search_decorator
    async def _execute_round_1(self, question, **kwargs):
        loop_indices = list(range(self.agents, self.agents * 2))
        random.shuffle(loop_indices)

        activated_indices = []
        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0

        for idx, node_idx in enumerate(loop_indices):
            await asyncio.to_thread(self.nodes[node_idx].activate, question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)

            if idx >= math.floor(2 / 3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    trace = "=== Round 1 (Consensus Reached) ===\n"
                    for a_idx in activated_indices:
                        trace += f"Agent {a_idx}: {self.nodes[a_idx].get_reply()}...\n"

                    return {
                        "response": trace,
                        "consensus_reached": True,
                        "consensus_reply": reply,
                        "activated_indices": activated_indices,
                        "resp_cnt": resp_cnt,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    }

        trace = "=== Round 1 ===\n"
        for a_idx in activated_indices:
            trace += f"Agent {a_idx}: {self.nodes[a_idx].get_reply()}...\n"

        return {
            "response": trace,
            "consensus_reached": False,
            "activated_indices": activated_indices,
            "resp_cnt": resp_cnt,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

    @llm_parallel_search_decorator
    async def _execute_round_n(self, question, rid, idxs_prev, **kwargs):
        idx_mask = list(range(self.agents))
        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0

        if self.agents > 3:
            replies = [self.nodes[idx].get_reply() for idx in idxs_prev]
            indices = list(range(len(replies)))
            random.shuffle(indices)
            shuffled_replies = [replies[idx] for idx in indices]

            tops, prompt_tokens, completion_tokens = await asyncio.to_thread(
                self.activation, shuffled_replies, question, self.qtype, self.mtype
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            idx_mask = list(map(lambda x: idxs_prev[indices[x]] % self.agents, tops))
            resp_cnt += self.activation_cost

        loop_indices = list(range(self.agents * rid, self.agents * (rid + 1)))
        random.shuffle(loop_indices)
        idxs = []

        for idx, node_idx in enumerate(loop_indices):
            if idx in idx_mask:
                await asyncio.to_thread(self.nodes[node_idx].activate, question)
                resp_cnt += 1
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens
                idxs.append(node_idx)

                if len(idxs) > math.floor(2 / 3 * len(idx_mask)):
                    reached, reply = self.check_consensus(idxs, idx_mask)
                    if reached:
                        trace = f"=== Round {rid} (Consensus Reached) ===\n"
                        for a_idx in idxs:
                            trace += f"Agent {a_idx}: {self.nodes[a_idx].get_reply()}...\n"

                        return {
                            "response": trace,
                            "consensus_reached": True,
                            "consensus_reply": reply,
                            "activated_indices": idxs,
                            "resp_cnt": resp_cnt,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                        }

        trace = f"=== Round {rid} ===\n"
        for a_idx in idxs:
            trace += f"Agent {a_idx}: {self.nodes[a_idx].get_reply()}...\n"

        return {
            "response": trace,
            "consensus_reached": False,
            "activated_indices": idxs,
            "resp_cnt": resp_cnt,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

    async def forward_async(self, question):
        def get_completions():
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2

        round0_result = await self._execute_round_0(
            question=question,
            task_type="qa",
        )

        resp_cnt += round0_result["resp_cnt"]
        total_prompt_tokens += round0_result["prompt_tokens"]
        total_completion_tokens += round0_result["completion_tokens"]

        if round0_result["consensus_reached"]:
            return (
                round0_result["consensus_reply"],
                resp_cnt,
                get_completions(),
                total_prompt_tokens,
                total_completion_tokens,
            )

        round1_result = await self._execute_round_1(
            question=question,
            task_type="qa",
        )

        resp_cnt += round1_result["resp_cnt"]
        total_prompt_tokens += round1_result["prompt_tokens"]
        total_completion_tokens += round1_result["completion_tokens"]

        if round1_result["consensus_reached"]:
            return (
                round1_result["consensus_reply"],
                resp_cnt,
                get_completions(),
                total_prompt_tokens,
                total_completion_tokens,
            )

        idxs = round1_result["activated_indices"]
        for rid in range(2, self.rounds):
            roundn_result = await self._execute_round_n(
                question=question,
                rid=rid,
                idxs_prev=idxs,
                task_type="qa",
            )

            resp_cnt += roundn_result["resp_cnt"]
            total_prompt_tokens += roundn_result["prompt_tokens"]
            total_completion_tokens += roundn_result["completion_tokens"]

            if roundn_result["consensus_reached"]:
                return (
                    roundn_result["consensus_reply"],
                    resp_cnt,
                    get_completions(),
                    total_prompt_tokens,
                    total_completion_tokens,
                )

            idxs = roundn_result["activated_indices"]

        completions = get_completions()
        result = most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0]
        return result, resp_cnt, completions, total_prompt_tokens, total_completion_tokens

    def forward(self, question):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.forward_async(question))


def set_rd_seed(seed):
    random.seed(seed)


def normalize_choice(answer):
    if answer is None:
        return ""
    return str(answer).strip().upper()


async def process_single_problem(que, ans, roles, model, activation, qtype):
    llmlp = LLMLP_GPQA(model, len(roles), roles, 3, activation, qtype, model)

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
    print(f"Processing {len(qa_pairs)} GPQA problems in async batches...")

    output_json_path = DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.json'
    output_txt_path = DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.txt'

    with open(output_json_path, 'w', encoding='utf-8') as f:
        f.write("")

    accs, resp_cnts, importances = [], 0, []
    total_prompt_tokens, total_completion_tokens = 0, 0

    for batch_idx in range(0, len(qa_pairs), ASYNC_BATCH_SIZE):
        batch = qa_pairs[batch_idx:batch_idx + ASYNC_BATCH_SIZE]
        print(f"Processing batch {batch_idx // ASYNC_BATCH_SIZE + 1}/{(len(qa_pairs)-1)//ASYNC_BATCH_SIZE + 1} ({len(batch)} problems)...")

        tasks = [
            process_single_problem(que, ans, ROLES, MODEL, ACTIVATION, TYPE)
            for que, ans in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Error processing problem: {result}")
                continue

            accs.append(result['acc'])
            resp_cnts += result['resp_cnt']
            importances.append(result['importance'])
            total_prompt_tokens += result['prompt_tokens']
            total_completion_tokens += result['completion_tokens']

            with open(output_json_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result['completion']) + '\n')

        if accs:
            print(f"Batch complete. Current accuracy: {sum(accs)/len(accs):.3f}")

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(str(accs) + ' ' + str(sum(accs) / len(qa_pairs) if qa_pairs else 0) + '\n')
        f.write(str(resp_cnts) + " " + str(resp_cnts / len(qa_pairs) if qa_pairs else 0) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos) / len(qa_pairs) for pos in zip(*importances)] if importances else []) + '\n')
        f.write(str(total_prompt_tokens) + ' ' + str(total_completion_tokens) + '\n')


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
