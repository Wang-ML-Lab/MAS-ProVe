import random
import re

from prompt_lib import ROLE_MAP, SYSTEM_PROMPT_GPQA, construct_ranking_message, construct_message
from utils import parse_single_choice, generate_answer


class LLMNeuron:

    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_single_choice, qtype="single_choice"):
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
            self.model = "chatgpt0301"
        elif "gpt" in mtype.lower():
            self.model = mtype
        else:
            raise NotImplementedError(f"Error init model type: {mtype}")

        def find_array(text):
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                last_match = matches[-1].replace(' ', '')
                def convert(x):
                    try:
                        return int(x)
                    except Exception:
                        return 0
                try:
                    return list(map(convert, last_match.split(',')))
                except Exception:
                    return []
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
        contexts, formers = self.get_context()
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]

        contexts.append(construct_message(formers, question, self.qtype))
        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
        # print(self.get_reply())
        self.answer = self.ans_parser(self.reply)
        weights = self.weights_parser(self.reply)
        if len(weights) != len(formers):
            # print("miss match!")
            weights = [0 for _ in range(len(formers))]

        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1

        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        # print(self.answer)
        # print([edge.weight for edge in self.from_edges])

    def get_context(self):
        if self.qtype == "single_choice":
            sys_prompt = ROLE_MAP[self.role] + "\n" + SYSTEM_PROMPT_GPQA
        else:
            raise NotImplementedError("Error init question type")
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


class LLMEdge:

    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0


def listwise_ranker_2(responses, question, qtype, model="chatgpt0301"):
    if model == "gpt-3.5-turbo":
        model = "chatgpt0301"
    elif model == "gpt-4":
        model = "gpt4"
    elif "gpt-5" in model.lower():
        pass
    else:
        print(f"Warning: Unknown model type '{model}', attempting to use directly")

    assert 2 < len(responses)
    message = construct_ranking_message(responses, question, qtype)
    completion, prompt_tokens, completion_tokens = generate_answer([message], model)

    pattern = r'\[([1234567]),\s*([1234567])\]'
    matches = re.findall(pattern, completion)
    try:
        match = matches[-1]
        tops = [int(match[0]) - 1, int(match[1]) - 1]
        def clip(x):
            if x < 0:
                return 0
            if x > len(responses) - 1:
                return len(responses) - 1
            return x
        tops = [clip(x) for x in tops]
    except Exception:
        print("error in parsing ranks")
        tops = random.sample(list(range(len(responses))), 2)

    return tops, prompt_tokens, completion_tokens