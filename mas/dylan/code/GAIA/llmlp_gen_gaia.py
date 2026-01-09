import ast
import json
import os
import random
import sys
import re
import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMLU'))
from LLMLP import LLMLP, ACTIVATION_MAP
from LLM_Neuron import LLMEdge, listwise_ranker_2
from llm_neuron_gaia import LLMNeuron_GAIA
from utils import *
from gaia_utils import parse_gaia_answer, gaia_string_match
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator

QUERY_DIR = sys.argv[1]  # Directory containing GAIA JSON files
MIN_FILE = int(sys.argv[2])  # Starting file number
MAX_FILE = int(sys.argv[3])  # Ending file number
EXP_NAME = sys.argv[4]  # Experiment name (e.g., "gaia_level1")
MODEL = sys.argv[5]  # Model name
DIR_NAME = sys.argv[6]  # Output directory base
ROLES = ast.literal_eval(sys.argv[7])  # Agent roles as list

ACTIVATION = "listwise"
TYPE = "gaia_qa"  # GAIA question-answering type
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)


class LLMLP_GAIA(LLMLP):
    """Extended LLMLP class for GAIA dataset with full DyLAN features"""
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=3, activation="listwise", qtype="gaia_qa", mtype="gpt-3.5-turbo"):
        # Store GAIA-specific settings before calling parent
        self.gaia_ans_parser = parse_gaia_answer
        self.gaia_cmp_res = gaia_string_match
        
        # Call parent constructor
        super().__init__(default_model_name, agents, agent_roles, rounds, activation, qtype, mtype)
        
        # Override with GAIA-specific functions after parent init
        self.cmp_res = gaia_string_match
        self.ans_parser = parse_gaia_answer
    
    def init_nn(self, activation, agent_roles):
        """Initialize neural network with GAIA-specific neurons"""
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron_GAIA(agent_roles[idx], self.mtype, parse_gaia_answer, self.qtype))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron_GAIA(agent_roles[idx], self.mtype, parse_gaia_answer, self.qtype))
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
        """Execute Round 0 - EXACT same logic as original, just extracted for RoundWise eval"""
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
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    trace = "=== Round 0 (Consensus Reached) ===\n"
                    for a_idx in activated_indices:
                        trace += f"Agent {a_idx} ({self.agent_roles[a_idx % len(self.agent_roles)]}): {self.nodes[a_idx].get_reply()[:200]}...\n"
                    
                    return {
                        "response": trace,
                        "consensus_reached": True,
                        "consensus_reply": reply,
                        "activated_indices": activated_indices,
                        "resp_cnt": resp_cnt,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens
                    }
        
        trace = "=== Round 0 ===\n"
        for a_idx in activated_indices:
            trace += f"Agent {a_idx} ({self.agent_roles[a_idx % len(self.agent_roles)]}): {self.nodes[a_idx].get_reply()[:200]}...\n"
        
        return {
            "response": trace,
            "consensus_reached": False,
            "activated_indices": activated_indices,
            "resp_cnt": resp_cnt,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }
    
    @llm_parallel_search_decorator
    async def _execute_round_1(self, question, **kwargs):
        """Execute Round 1 - EXACT same logic as original"""
        loop_indices = list(range(self.agents, self.agents*2))
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
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    trace = "=== Round 1 (Consensus Reached) ===\n"
                    for a_idx in activated_indices:
                        trace += f"Agent {a_idx} ({self.agent_roles[a_idx % len(self.agent_roles)]}): {self.nodes[a_idx].get_reply()[:200]}...\n"
                    
                    return {
                        "response": trace,
                        "consensus_reached": True,
                        "consensus_reply": reply,
                        "activated_indices": activated_indices,
                        "resp_cnt": resp_cnt,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens
                    }
        
        trace = "=== Round 1 ===\n"
        for a_idx in activated_indices:
            trace += f"Agent {a_idx} ({self.agent_roles[a_idx % len(self.agent_roles)]}): {self.nodes[a_idx].get_reply()[:200]}...\n"
        
        return {
            "response": trace,
            "consensus_reached": False,
            "activated_indices": activated_indices,
            "resp_cnt": resp_cnt,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }
    
    @llm_parallel_search_decorator
    async def _execute_round_n(self, question, rid, idxs_prev, **kwargs):
        """Execute Round N (2+) - EXACT same logic as original"""
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

        loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
        random.shuffle(loop_indices)
        idxs = []
        
        for idx, node_idx in enumerate(loop_indices):
            if idx in idx_mask:
                await asyncio.to_thread(self.nodes[node_idx].activate, question)
                resp_cnt += 1
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens
                idxs.append(node_idx)
                
                if len(idxs) > math.floor(2/3 * len(idx_mask)):
                    reached, reply = self.check_consensus(idxs, idx_mask)
                    if reached:
                        trace = f"=== Round {rid} (Consensus Reached) ===\n"
                        for a_idx in idxs:
                            trace += f"Agent {a_idx} ({self.agent_roles[a_idx % len(self.agent_roles)]}): {self.nodes[a_idx].get_reply()[:200]}...\n"
                        
                        return {
                            "response": trace,
                            "consensus_reached": True,
                            "consensus_reply": reply,
                            "activated_indices": idxs,
                            "resp_cnt": resp_cnt,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens
                        }
        
        trace = f"=== Round {rid} ===\n"
        for a_idx in idxs:
            trace += f"Agent {a_idx} ({self.agent_roles[a_idx % len(self.agent_roles)]}): {self.nodes[a_idx].get_reply()[:200]}...\n"
        
        return {
            "response": trace,
            "consensus_reached": False,
            "activated_indices": idxs,
            "resp_cnt": resp_cnt,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }
    
    async def forward_async(self, question):
        """Async forward using decorated rounds - SAME FLOW as original LLMLP.forward()"""
        def get_completions():
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2

        # Round 0 - decorated (runs 3x, judge picks best)
        round0_result = await self._execute_round_0(
            question=question,
            task_type=self.qtype
        )
        
        resp_cnt += round0_result["resp_cnt"]
        total_prompt_tokens += round0_result["prompt_tokens"]
        total_completion_tokens += round0_result["completion_tokens"]
        
        if round0_result["consensus_reached"]:
            return (round0_result["consensus_reply"], resp_cnt, get_completions(), 
                   total_prompt_tokens, total_completion_tokens)

        # Round 1 - decorated
        round1_result = await self._execute_round_1(
            question=question,
            task_type=self.qtype
        )
        
        resp_cnt += round1_result["resp_cnt"]
        total_prompt_tokens += round1_result["prompt_tokens"]
        total_completion_tokens += round1_result["completion_tokens"]
        
        if round1_result["consensus_reached"]:
            return (round1_result["consensus_reply"], resp_cnt, get_completions(),
                   total_prompt_tokens, total_completion_tokens)

        # Rounds 2+ - decorated
        idxs = round1_result["activated_indices"]
        for rid in range(2, self.rounds):
            roundn_result = await self._execute_round_n(
                question=question,
                rid=rid,
                idxs_prev=idxs,
                task_type=self.qtype
            )
            
            resp_cnt += roundn_result["resp_cnt"]
            total_prompt_tokens += roundn_result["prompt_tokens"]
            total_completion_tokens += roundn_result["completion_tokens"]
            
            if roundn_result["consensus_reached"]:
                return (roundn_result["consensus_reply"], resp_cnt, get_completions(),
                       total_prompt_tokens, total_completion_tokens)
            
            idxs = roundn_result["activated_indices"]

        # Final result - same as original
        completions = get_completions()
        result = most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0]
        print(f"[LLMLP_GAIA] Most frequent answer: {result}")
        return result, resp_cnt, completions, total_prompt_tokens, total_completion_tokens
    
    def forward(self, question):
        """Synchronous wrapper - maintains original interface"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.forward_async(question))


def set_rd_seed(seed):
    random.seed(seed)


def get_gaia_qa_pairs(query_dir, min_file, max_file):
    """Load GAIA question-answer pairs from individual JSON files"""
    ret_list = []
    for subdir, dirs, files in os.walk(query_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            
            filename_without_ext = os.path.splitext(file)[0]
            if not filename_without_ext.isdigit():
                continue
                
            file_num = int(filename_without_ext)
            if min_file <= file_num <= max_file:
                with open(os.path.join(subdir, file), 'r') as fp:
                    try:
                        problem_data = json.load(fp)
                    except Exception as e:
                        print(f"Error loading JSON from {file}", e)
                        continue
                    
                    question = problem_data["problem"]
                    answer = str(problem_data['answer']).strip()
                    ret_list.append((question, answer))
    
    return sorted(ret_list, key=lambda x: x[0])  # Sort for consistency


async def process_single_problem(que, ans, roles, model, activation, qtype):
    """Process a single problem asynchronously"""
    llmlp = LLMLP_GAIA(model, len(roles), roles, 3, activation, qtype, model)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        llmlp.zero_grad()
        res, resp_cnt, completions, prompt_tokens, completion_tokens = await loop.run_in_executor(
            executor, llmlp.forward, que
        )
        imp_score = await loop.run_in_executor(
            executor, llmlp.backward, res
        )

    return {
        'completion': completions,
        'acc': gaia_string_match(ans, res),
        'resp_cnt': resp_cnt,
        'importance': imp_score,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens
    }


async def main_async():
    set_rd_seed(0)
    assert len(ROLES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    qa_pairs = get_gaia_qa_pairs(QUERY_DIR, MIN_FILE, MAX_FILE)
    print(f"Processing {len(qa_pairs)} GAIA problems in async batches...")

    accs, resp_cnts, importances = [], 0, []
    completion_list = []
    total_prompt_tokens, total_completion_tokens = 0, 0

    batch_size = 50
    all_results = []

    for batch_idx in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[batch_idx:batch_idx + batch_size]
        print(f"Processing batch {batch_idx//batch_size + 1}/{(len(qa_pairs)-1)//batch_size + 1} ({len(batch)} problems)...")

        tasks = [process_single_problem(que, ans, ROLES, MODEL, ACTIVATION, TYPE) for que, ans in batch]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:

            import traceback
            if isinstance(result, Exception):
                print(f"Error processing problem: {result}")
                traceback.print_exception(type(result), result, result.__traceback__)
                continue

            all_results.append(result)
            completion_list.append(result['completion'])
            accs.append(result['acc'])
            resp_cnts += result['resp_cnt']
            importances.append(result['importance'])
            total_prompt_tokens += result['prompt_tokens']
            total_completion_tokens += result['completion_tokens']
            # Save incrementally (match AIME DyLAN logging: newline-delimited JSON per completion)
            with open(DIR_NAME + '/' + EXP_NAME + '.json', 'a') as f:
                f.write(json.dumps(result['completion']) + '\n')

        print(f"Batch {batch_idx//batch_size + 1} complete. Current accuracy: {sum(accs)/len(accs) if accs else 0:.3f}")

    print(accs)
    print(resp_cnts)
    print(importances)
    # Write summary text file (DyLAN text format)
    with open(os.path.join(DIR_NAME, EXP_NAME + '.txt'), 'w') as f:
        f.write(str(accs) + ' ' + str(sum(accs)/len(qa_pairs) if qa_pairs else 0) + '\n')
        f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs) if qa_pairs else 0) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos)/len(qa_pairs) for pos in zip(*importances)] if importances else []) + '\n')
        f.write(str(total_prompt_tokens) + '\n')
        f.write(str(total_completion_tokens) + '\n')


def main():
    """Synchronous wrapper for async main"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
