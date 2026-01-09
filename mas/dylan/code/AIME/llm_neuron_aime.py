"""
AIME-specific LLM Neuron implementation (follows GAIA structure exactly)
"""
import random
import re
import sys
import os
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (crucial for sync wrappers in threaded apps)
nest_asyncio.apply()
from prompt_lib_aime import ROLE_MAP_AIME, SYSTEM_PROMPT_AIME
from aime_utils import extract_math_answer, generate_answer_aime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMLU'))
from LLM_Neuron import LLMEdge
from prompt_lib import construct_message

class LLMNeuron_AIME:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=extract_math_answer, qtype="aime_qa"):
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
                    try:
                        return int(x)
                    except Exception:
                        return 0
                try:
                    ret = list(map(convert, last_match.split(',')))
                except Exception:
                    ret = []
                return ret
            else:
                return []
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
        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer_aime(contexts, self.model)
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
        sys_prompt = ROLE_MAP_AIME.get(self.role, ROLE_MAP_AIME["Assistant"]) + "\n" + SYSTEM_PROMPT_AIME
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