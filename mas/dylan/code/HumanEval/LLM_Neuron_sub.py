"""
HumanEval-specific LLM Neuron implementation with parallel search decorator (follows AIME pattern)
"""
import random
import re
import sys
import os
import asyncio
import math
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (crucial for sync wrappers in threaded apps)
nest_asyncio.apply()

from utils import parse_code_completion, generate_answer, parse_judge_attitude
from prompt_lib import *
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator


class LLMNeuron:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_code_completion, qtype="single_choice"):
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

        if mtype == "gpt-3.5-turbo":
            self.model = "gpt-3.5-turbo-0301"
        elif "gpt" in mtype.lower():
            # Support any GPT model (gpt-4, gpt-5-mini, etc.)
            self.model = mtype
        else:
            raise NotImplementedError("Error init model type")

        def find_array(text):
            # Find all matches of array pattern
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                # Take the last match and remove spaces
                last_match = matches[-1].replace(' ', '')
                # Convert the string to a list of integers
                try:
                    ret = list(map(int, last_match.split(',')))
                except:
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

    # Async Decorated Method for Process Evaluation (AgentWise)
    @llm_parallel_search_decorator
    async def _activate_async(self, question=None, **kwargs):
        """
        Runs in parallel via decorator. Returns a DICT so the decorator 
        can send 'response' (reasoning) to the Judge and return 'answer' to us.
        """
        # Note: 'task_type' is popped by the decorator, but we use self.qtype anyway.
        if question is None:
            question = self.question

        # 1. Prepare Context (Standard HumanEval logic)
        contexts, formers = self.get_context()
        
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers_content = [mess[0] for mess in formers]

        contexts.append(construct_message(formers_content, question, self.qtype))
        
        # 2. Generate (Run in thread to avoid blocking asyncio loop)
        reply, p_tokens, c_tokens = await asyncio.to_thread(
            generate_answer, contexts, self.model
        )
        
        # 3. Local calculation (Do NOT update self state here to avoid race conditions)
        answer = self.ans_parser(reply, question)
        weights = self.weights_parser(reply)
        
        # Handle Ranker special case (same as base)
        if len(weights) != len(formers_content) - 1:
            weights = [0 for _ in range(len(formers_content))]
        else:
            res_weights = []
            if formers_content[0].role == "Ranker":
                res_weights.append(3)
            for wid, weight in enumerate(weights):
                res_weights.append(weight)
                if wid + 1 < len(formers_content) and formers_content[wid + 1].role == "Ranker":
                    res_weights.append(3)
            weights = res_weights

        # Re-sort weights to match original edge order
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers_content))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        sorted_weights = [weight for _, weight, _ in sorted_pairs]
        sorted_edge_ids = [eid for eid, _, _ in sorted_pairs]

        # 4. Return structured data
        # The decorator's flatten_to_str will find 'response' for the Judge.
        return {
            "response": reply,        # reasoning trace for Judge
            "answer": answer,         # final answer for Agent
            "weights": sorted_weights,
            "edge_ids": sorted_edge_ids,
            "p_tokens": p_tokens,
            "c_tokens": c_tokens,
        }

    # -------------------------------------------------------------------------
    # Synchronous Wrapper
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
            winning_edge_ids = best_result.get("edge_ids", [])
        else:
            # Fallback if something fails and returns a string
            self.reply = str(best_result)
            self.answer = self.ans_parser(self.reply, self.question)
            winning_weights = []
            winning_edge_ids = []

        # Update Edge Weights (using the winner's weights)
        formers = [(edge.a1, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        former_eids = [eid for _, eid in formers]
        
        if len(winning_weights) == len(former_eids) and len(winning_edge_ids) == len(former_eids):
            for eid, weight in zip(winning_edge_ids, winning_weights):
                self.from_edges[eid].weight = weight / 5 if 0 < weight <= 5 else (1 if weight > 5 else 0)
            
            # Normalize
            total = sum([self.from_edges[eid].weight for eid in winning_edge_ids])
            if total > 0:
                for eid in winning_edge_ids:
                    self.from_edges[eid].weight /= total
            else:
                for eid in winning_edge_ids:
                    self.from_edges[eid].weight = 1 / len(winning_edge_ids)

        print(f"Agent Answer: {self.answer}")
        print([edge.weight for edge in self.from_edges])
    
    def get_context(self):
        if self.qtype == "single_choice":
            sys_prompt = ROLE_MAP[self.role] + "\n" + SYSTEM_PROMPT_MMLU
        elif self.qtype == "code_completion":
            if len(self.from_edges) == 0:
                sys_prompt = ROLE_MAP_INIT[self.role]
            else:
                sys_prompt = ROLE_MAP[self.role]
        else:
            raise NotImplementedError("Error init question type")
        
        contexts = [{"role": "system", "content": sys_prompt}]
        
        formers = [(edge.a1, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class JudgeNeuron:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_judge_attitude, qtype="single_choice"):
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
        self.resp_cost = 1 if role not in TOOL_LIST else 0

        if mtype == "gpt-3.5-turbo":
            self.model = "gpt-3.5-turbo-0301"
        elif "gpt" in mtype.lower():
            # Support any GPT model (gpt-4, gpt-5-mini, etc.)
            self.model = mtype
        else:
            raise NotImplementedError("Error init model type")

        def find_array(text, formers):
            if self.role == "Ranker":
                results = []
                for former in formers:
                    if self.answer[former]:
                        results.append(5)
                    else:
                        results.append(0)
                return results
            if self.role == "Passer":
                results = []
                for former in formers:
                    if self.answer[former] == "The code doesn't have syntax error.":
                        results.append(5)
                    else:
                        results.append(0)
                return results
            if self.role == "Tester":
                results = []
                for former in formers:
                    if "Passed tests:\n" not in self.answer[former]:
                        results.append(0)
                    else:
                        total_tests = len(self.answer[former].splitlines()) - 1
                        if "Failed tests:\n" in self.answer[former]:
                            total_tests -= 2
                        flag_pass = False
                        pass_tests = 0
                        for line in self.answer[former].splitlines():
                            if flag_pass:
                                if "Failed tests:" in line:
                                    break
                                pass_tests += 1
                            if "Passed tests:" in line:
                                flag_pass = True
                        pass_tests -= 1
                        results.append(math.ceil(pass_tests / total_tests * 5))
                return results

            if self.role != "Reflector" and self.role != "Debugger" and self.role != "QualityManager":
                raise NotImplementedError("Error init role type")

            # Find all matches of array pattern
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                # Take the last match and remove spaces
                last_match = matches[-1].replace(' ', '')
                # Convert the string to a list of integers
                try:
                    ret = list(map(int, last_match.split(',')))
                except:
                    ret = []
                return ret
            else:
                return []
        self.weights_parser = find_array

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unit_tests = []

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer
    
    def get_unit_tests(self):
        return self.unit_tests

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unit_tests = []

    def activate(self, question):
        self.question = question
        self.active = True
        # get context and genrate reply
        contexts, formers = self.get_context()
        # print("formers: ", formers)
        # shuffle
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]
        # print("shuffled: ", shuffled_idxs)

        if self.role == "Ranker" and len(formers) <= 2:
            self.reply = "[1, 2]"
            print(self.get_reply())

        elif self.role not in TOOL_LIST:
            contexts.append(construct_judge_message(formers, question, self.qtype, self.role))
            print(contexts)
            self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
            print(self.get_reply())
        else:
            self.reply = formers
            print(self.get_reply())

        # parse answer
        self.answer = self.ans_parser(self.reply, question, self.role, formers)
        if self.role == "Tester":
            self.answer, self.unit_tests = self.answer

        weights = self.weights_parser(self.reply, formers)
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]
        
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        # normalize weights
        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        print(self.answer)

        
    def get_context(self):
        if self.qtype == "code_completion":
            sys_prompt = JUDGE_MAP[self.role]
        else:
            raise NotImplementedError("Error init question type")
        
        contexts = [{"role": "system", "content": sys_prompt}]
        
        formers = [(edge.a1.answer, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class LLMEdge:

    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0
