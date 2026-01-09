import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer


ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1} # TODO: only 0 is implemented

class LLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo"):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        # print(f"[LLMLP] Initializing {self.agents} agents for {self.rounds} rounds.")
        # Initialize the first round of agents (neurons)
        for idx in range(self.agents):
            # print(f"[LLMLP] Creating agent {idx} with role {agent_roles[idx]}")
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
        
        agents_last_round = self.nodes[:self.agents]
        # For each subsequent round, create new agents and connect edges from previous round
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                # print(f"[LLMLP] Creating agent {idx} for round {rid} with role {agent_roles[idx]}")
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        # Set activation function and cost
        if activation == 0:
            print("[LLMLP] Using listwise_ranker_2 as activation function.")
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")
    
    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        # Check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        # print(f"[LLMLP] Checking consensus among candidates: {candidates}")
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        # If more than 2/3 of the agents agree, consensus is reached
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            # print(f"[LLMLP] Consensus answer: {consensus_answer} (count: {ca_cnt})")
            return True, consensus_answer
        # print("[LLMLP] No consensus reached.")
        return False, None

    def set_allnodes_deactivated(self):
        # Deactivate all nodes before a new forward pass
        # print("[LLMLP] Deactivating all nodes.")
        for node in self.nodes:
            node.deactivate()

    def forward(self, question):
        def get_completions():
            # Gather completions for each agent in each round
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            # print(f"[LLMLP] Completions: {completions}")
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2
        # question = format_question(question, self.qtype)

        # Shuffle the order of agents for the first round
        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            # print(f"[LLMLP] Activating agent {node_idx} in round 0 (shuffled idx {idx})")
            self.nodes[node_idx].activate(question)
            # print(f"[LLMLP] Agent {node_idx} reply: {self.nodes[node_idx].get_reply()}")
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            # Uncomment to enable early consensus check after each activation
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        # Shuffle the order of agents for the second round
        loop_indices = list(range(self.agents, self.agents*2))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            # print(f"[LLMLP] Activating agent {node_idx} in round 1 (shuffled idx {idx})")
            self.nodes[node_idx].activate(question)
            # print(f"[LLMLP] Agent {node_idx} reply: {self.nodes[node_idx].get_reply()}")
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            # Uncomment to enable early consensus check after each activation
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        # For subsequent rounds, use activation and mask logic
        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents*2))
        for rid in range(2, self.rounds):
            # TODO: Make compatible with 1/2 agents
            if self.agents > 3:
                # Gather replies and shuffle for activation function
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                # print(f"[LLMLP] Round {rid} replies before shuffle: {replies}")
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
                # print(f"[LLMLP] Round {rid} shuffled replies: {shuffled_replies}")
                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype)
                # print(f"[LLMLP] Activation function tops: {tops}")
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents, tops))
                # print(f"[LLMLP] idx_mask after activation: {idx_mask}")
                resp_cnt += self.activation_cost

            # Shuffle the order of agents for this round
            loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    print(f"[LLMLP] Activating agent {node_idx} in round {rid} (shuffled idx {idx})")
                    self.nodes[node_idx].activate(question)
                    print(f"[LLMLP] Agent {node_idx} reply: {self.nodes[node_idx].get_reply()}")
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    # Uncomment to enable early consensus check after each activation
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        completions = get_completions()
        # print(f"[LLMLP] Final completions: {completions}")
        # Return the most frequent answer among the final active nodes, along with stats
        result = most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0]
        print(f"[LLMLP] Most frequent answer: {result}")
        return result, resp_cnt, completions, total_prompt_tokens, total_completion_tokens


    def backward(self, result):
        # Backward pass to assign importance to nodes based on final result
        # print(f"[LLMLP] Starting backward pass with result: {result}")
        flag_last = False
        for rid in range(self.rounds-1, -1, -1):
            # print(f"[LLMLP] Backward round {rid}")
            if not flag_last:
                # Only process the last round with active nodes
                active_idxs = [idx for idx in range(self.agents*rid, self.agents*(rid+1)) if self.nodes[idx].active]
                # print(f"[LLMLP] Active indices in last round: {active_idxs}")
                if len(active_idxs) > 0:
                    flag_last = True
                else:
                    continue

                # Assign equal importance to all active nodes with correct answer
                correct_idxs = [idx for idx in active_idxs if self.cmp_res(self.nodes[idx].get_answer(), result)]
                # print(f"[LLMLP] Correct indices in last round: {correct_idxs}")
                if len(correct_idxs) > 0:
                    ave_w = 1 / len(correct_idxs)
                else:
                    ave_w = 0
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result):
                        self.nodes[idx].importance = ave_w
                    else:
                        self.nodes[idx].importance = 0
            else:
                # For earlier rounds, propagate importance backward through edges
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance
            # print(f"[LLMLP] Importances after round {rid}: {[self.nodes[idx].importance for idx in range(self.agents*rid, self.agents*(rid+1))]}")

        # Return the importance of all nodes after backward pass
        all_importances = [node.importance for node in self.nodes]
        # print(f"[LLMLP] Final node importances: {all_importances}")
        return all_importances
