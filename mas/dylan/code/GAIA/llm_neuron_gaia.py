"""
GAIA-specific LLM Neuron implementation
"""
import random
import re
import sys
import os
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()
from prompt_lib_gaia import ROLE_MAP_GAIA, SYSTEM_PROMPT_GAIA
from gaia_utils import parse_gaia_answer, generate_answer_with_tools
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMLU'))
from LLM_Neuron import LLMEdge
from prompt_lib import construct_message
# Import the universal decorator

class LLMNeuron_GAIA:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_gaia_answer, qtype="gaia_qa"):
        self.role = role
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None

        if mtype == "gpt-3.5-turbo" or "gpt" in mtype.lower():
            self.model = mtype
        else:
            raise NotImplementedError("Error init model type")

        def find_array(text):
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                last_match = matches[-1].replace(' ', '')
                def convert(x):
                    try: return int(x)
                    except: return 0
                try: ret = list(map(convert, last_match.split(',')))
                except: ret = []
                return ret
            else: return []
        self.weights_parser = find_array

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -------------------------------------------------------------------------
    # NEW: Async Decorated Method for Process Evaluation
    # -------------------------------------------------------------------------
    # @llm_parallel_search_decorator
    # async def _activate_async(self, question=None, **kwargs):
    #     """
    #     Runs in parallel via decorator. Returns a DICT so the decorator 
    #     can send 'response' (reasoning) to the Judge and return 'answer' to us.
    #     """
    #     if question is None:
    #         question = self.question

    #     # 1. Prepare Context
    #     contexts, formers = self.get_context()
        
    #     original_idxs = [mess[1] for mess in formers]
    #     random.shuffle(formers)
    #     shuffled_idxs = [mess[1] for mess in formers]
    #     formers_content = [mess[0] for mess in formers]

    #     contexts.append(construct_message(formers_content, question, self.qtype))
        
    #     # 2. Generate (Run in thread to avoid blocking asyncio loop)
    #     # Note: generate_answer_with_tools is likely heavy, so threading is crucial
    #     reply, p_tokens, c_tokens = await asyncio.to_thread(
    #         generate_answer_with_tools, contexts, self.model
    #     )
        
    #     # 3. Local calculation
    #     answer = self.ans_parser(reply)
    #     weights = self.weights_parser(reply)
        
    #     if len(weights) != len(formers):
    #         weights = [0 for _ in range(len(formers))]

    #     # Re-sort weights
    #     shuffled_pairs = list(zip(shuffled_idxs, weights))
    #     sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
    #     sorted_weights = [weight for _, weight in sorted_pairs]

    #     # 4. Return structured data
    #     return {
    #         "response": reply,        # reasoning trace for Judge
    #         "answer": answer,         # final answer for Agent
    #         "weights": sorted_weights,# edge weights for Agent
    #         "p_tokens": p_tokens,
    #         "c_tokens": c_tokens
    #     }

    # # -------------------------------------------------------------------------
    # # UPDATED: Synchronous Wrapper
    # # -------------------------------------------------------------------------
    # def activate(self, question):
    #     self.question = question
    #     self.active = True
        
    #     # Run the decorated async search
    #     try:
    #         loop = asyncio.get_running_loop()
    #     except RuntimeError:
    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)
            
    #     # PASS 'task_type' HERE
    #     best_result = loop.run_until_complete(
    #         self._activate_async(question=question, task_type=self.qtype)
    #     )
        
    #     # Unpack the "Winner"
    #     if isinstance(best_result, dict):
    #         self.reply = best_result.get("response", "")
    #         self.answer = best_result.get("answer", "")
    #         self.prompt_tokens = best_result.get("p_tokens", 0)
    #         self.completion_tokens = best_result.get("c_tokens", 0)
    #         winning_weights = best_result.get("weights", [])
    #     else:
    #         self.reply = str(best_result)
    #         self.answer = self.ans_parser(self.reply)
    #         winning_weights = []

    #     # Update Edge Weights
    #     active_edges = [edge for edge in self.from_edges if edge.a1.reply is not None and edge.a1.active]
        
    #     if len(winning_weights) == len(active_edges):
    #         lp = 0
    #         for edge in active_edges:
    #             w = winning_weights[lp]
    #             edge.weight = w / 5 if 0 < w <= 5 else (1 if w > 5 else 0)
    #             lp += 1
            
    #         # Normalize
    #         total = sum([edge.weight for edge in active_edges])
    #         if total > 0:
    #             for edge in active_edges:
    #                 edge.weight /= total
    #         else:
    #             for edge in active_edges:
    #                 edge.weight = 1 / len(active_edges)

    #     print(f"Winner Answer: {self.answer}")
    #     print([edge.weight for edge in self.from_edges])
    
    def activate(self, question):
        self.question = question
        self.active = True
        
        # Get context and generate reply
        contexts, formers = self.get_context()
        
        # Shuffle formers
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]

        contexts.append(construct_message(formers, question, self.qtype))
        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer_with_tools(contexts, self.model)
        print(self.get_reply())
        
        # Parse answer
        self.answer = self.ans_parser(self.reply)
        weights = self.weights_parser(self.reply)
        
        if len(weights) != len(formers):
            print("miss match!")
            weights = [0 for _ in range(len(formers))]

        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        print([self.from_edges[eid].weight for _, eid in formers])
        
        # Normalize weights
        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        print(self.answer)
        print([edge.weight for edge in self.from_edges])
        
    def get_context(self):
        sys_prompt = ROLE_MAP_GAIA.get(self.role, ROLE_MAP_GAIA["Assistant"]) + "\n" + SYSTEM_PROMPT_GAIA
        contexts = [{"role": "system", "content": sys_prompt}]
        
        formers = [(edge.a1.reply, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts