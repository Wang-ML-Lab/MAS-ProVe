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
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator
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

    # -------------------------------------------------------------------------
    # NEW: Async Decorated Method for Process Evaluation
    # -------------------------------------------------------------------------
    @llm_parallel_search_decorator
    async def _activate_async(self, question=None, **kwargs):
        """
        Runs in parallel via decorator. Returns a DICT so the decorator 
        can send 'response' (reasoning) to the Judge and return 'answer' to us.
        """
        # Note: 'task_type' is popped by the decorator, but we use self.qtype anyway.
        if question is None:
            question = self.question

        # 1. Prepare Context (Standard logic)
        contexts, formers = self.get_context()
        
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers_content = [mess[0] for mess in formers]

        contexts.append(construct_message(formers_content, question, self.qtype))
        context_str = ""
        for msg in contexts:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')
            context_str += f"{role}: {content}\n\n"
        # 2. Generate (Run in thread to avoid blocking asyncio loop)
        reply, p_tokens, c_tokens = await asyncio.to_thread(
            generate_answer_aime, contexts, self.model
        )
        
        # 3. Local calculation (Do NOT update self state here to avoid race conditions)
        answer = self.ans_parser(reply)
        weights = self.weights_parser(reply)
        
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]

        # Re-sort weights to match original edge order
        shuffled_pairs = list(zip(shuffled_idxs, weights))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        sorted_weights = [weight for _, weight in sorted_pairs]

        # 4. Return structured data
        # The decorator's flatten_to_str will find 'response' for the Judge.
        return {
            "response": reply,        # reasoning trace for Judge
            "answer": answer,         # final answer for Agent
            "weights": sorted_weights,# edge weights for Agent
            "p_tokens": p_tokens,
            "c_tokens": c_tokens,
            "context": context_str   # full context for debugging
        }

    # -------------------------------------------------------------------------
    # UPDATED: Synchronous Wrapper
    # -------------------------------------------------------------------------
    def activate(self, question):
        self.question = question
        self.active = True
        
        # Run the decorated async search
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # PASS 'task_type' HERE so the decorator (wrapper) sees it!
        best_result = loop.run_until_complete(
            self._activate_async(question=question, task_type=self.qtype)
        )
        
        # Unpack the "Winner"
        if isinstance(best_result, dict):
            self.reply = best_result.get("response", "")
            self.answer = best_result.get("answer", "")
            self.prompt_tokens = best_result.get("p_tokens", 0)
            self.completion_tokens = best_result.get("c_tokens", 0)
            winning_weights = best_result.get("weights", [])
        else:
            # Fallback if something fails and returns a string
            self.reply = str(best_result)
            self.answer = self.ans_parser(self.reply)
            winning_weights = []

        # Update Edge Weights (using the winner's weights)
        active_edges = [edge for edge in self.from_edges if edge.a1.reply is not None and edge.a1.active]
        
        if len(winning_weights) == len(active_edges):
            lp = 0
            for edge in active_edges:
                w = winning_weights[lp]
                edge.weight = w / 5 if 0 < w <= 5 else (1 if w > 5 else 0)
                lp += 1
            
            # Normalize
            total = sum([edge.weight for edge in active_edges])
            if total > 0:
                for edge in active_edges:
                    edge.weight /= total
            else:
                for edge in active_edges:
                    edge.weight = 1 / len(active_edges)

        print(f"Winner Answer: {self.answer}")
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