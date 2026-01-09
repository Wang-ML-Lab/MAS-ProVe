import ast
import json
import os
import random
import sys
import re
import asyncio
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

@llm_parallel_search_decorator
async def process_single_problem(que, ans, roles, model, activation, qtype, **kwargs):
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
    # 2. Construct the Trace String (SMARTER VERSION)
    trace_lines = []
    num_agents = len(roles)
    
    try:
        # completions is [ [R0, R1, R2], [R0, R1, R2] ... ]
        # We need to know how many rounds actually occurred.
        # Assuming all agents have lists of the same length (padded with None if needed).
        max_rounds = len(completions[0]) if completions else 0
        
        for r in range(max_rounds):
            # CHECK: Did any agent speak in this round?
            # If all agents returned None or empty string for this round, we STOP recording.
            active_responses = [completions[a_idx][r] for a_idx in range(num_agents)]
            if not any(active_responses):
                break # Stop processing empty rounds (Enough thinking!)

            trace_lines.append(f"--- Round {r} ---")
            for a_idx in range(num_agents):
                reply = active_responses[a_idx]
                if reply:
                    # Truncate slightly to keep Judge context manageable
                    clean_reply = reply.strip()
                    trace_lines.append(f"Agent {a_idx} ({roles[a_idx]}): {clean_reply}...")
            
            trace_lines.append("") # Spacer between rounds

    except Exception as e:
        trace_lines.append(f"Error parsing trace: {str(e)}")
        trace_lines.append(str(completions))

    trace_str = "\n".join(trace_lines) + f"\n\nFinal Answer: {res}"

    # 3. Return the Dictionary required by the Decorator/Judge
    # Keys 'response' and 'output' are mandatory for the parallel search logic
    return {
        'response': trace_str,      # The REASONING trace (Primary for Judge)
        'output': res,              # The actual return value
        'completion': completions,
        'acc': gaia_string_match(ans, res),
        'resp_cnt': resp_cnt,
        'importance': imp_score,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens
    }

# async def process_single_problem(que, ans, roles, model, activation, qtype):
#     """Process a single problem asynchronously"""
#     llmlp = LLMLP_GAIA(model, len(roles), roles, 3, activation, qtype, model)
#     loop = asyncio.get_event_loop()
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         llmlp.zero_grad()
#         res, resp_cnt, completions, prompt_tokens, completion_tokens = await loop.run_in_executor(
#             executor, llmlp.forward, que
#         )
#         imp_score = await loop.run_in_executor(
#             executor, llmlp.backward, res
#         )

#     return {
#         'completion': completions,
#         'acc': gaia_string_match(ans, res),
#         'resp_cnt': resp_cnt,
#         'importance': imp_score,
#         'prompt_tokens': prompt_tokens,
#         'completion_tokens': completion_tokens
#     }


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

        # tasks = [process_single_problem(que, ans, ROLES, MODEL, ACTIVATION, TYPE) for que, ans in batch]
        tasks = [process_single_problem(que, ans, ROLES, MODEL, ACTIVATION, TYPE,  question= que, task_type = TYPE) for que, ans in batch]
        
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
