import argparse
import json
import re
import time
from sglang import function, gen, RuntimeEndpoint
import fcntl
import os
import math
import yaml
import torch
import torch.nn.functional as F
import threading



def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_prompts(args):
    test_cases = read_jsonl(args.input_path)
    prompts = []
    for test in test_cases:
        prompts.append(test["problem"])
    return prompts, test_cases



class TreeNode:
    def __init__(self, id, state, score, num_step_tokens=0, parent=None):
        self.id = id
        self.state = state
        self.text_ = state.text()
        self.score_ = score
        self.parent = parent
        self.leaf_ = False
        self.cum_tokens = 0
        self.num_step_tokens = 0
        if parent is not None and "The answer is" in self.text_:
            self.leaf_ = True
        if parent is not None:
            self.depth = parent.get_depth() + 1
            self.cum_tokens += num_step_tokens
        else:
            self.depth = 0
            self.cum_tokens = num_step_tokens
     
    def get_id(self):
        return self.id
    
    def get_parent(self):
        return self.parent
    
    def get_text(self):
        return self.text_

    def get_state(self):
        return self.state
    
    def get_depth(self):
        return self.depth
    
    def get_score(self):
        return self.score_
    
    def is_leaf(self):
        return self.leaf_
    
    def get_cum_tokens(self):
        return self.cum_tokens

class Tree:
    def __init__(self, root_state, paras, reward_backend):
        self.size_ = 1
        self.nodes = []
        self.paras = paras
        self.reward_backend = reward_backend
        self.root_ = TreeNode(0,root_state, 1.0)
        self.remaining_width = paras["width"]
        self.history_list = []
        self.running_list = []
        self.depth_nodes = [[] for i in range(100)]
        self.nodes.append(self.root_)
        self.depth_nodes[0].append(self.root_)
    
    def reset_running_list(self):
        self.running_list = []
    
    def get_running_list(self):
        return self.running_list
    
    def get_history_list(self):
        return self.history_list
    
    def get_nodes(self):
        return self.nodes
    
    def expand(self, node, wid):
        state = node.get_state()
        forks = state.fork(wid)
        depth = node.get_depth()
        for fork in forks:
            fork.set_score_backend(self.reward_backend)
            if self.paras["policy_model_type"] == "mistral" or self.paras["policy_model_type"] == "llemma":
                fork += gen("step", self.paras["max_step_tokens"], stop="Step "+str(depth+2), temperature=self.paras["temperature"])
                fork += gen("score", max_tokens=0, forward_only=True, logits_require_id=8094)
            self.running_list.append((fork, node))
            self.history_list.append(fork)
    
    def insert(self, state, parent):
        if state.scores() == [] or state.scores == None:
            return
        score = state.scores()[-1]
        num_step_tokens = state.get_meta_info("step")["completion_tokens"]
        new_node = TreeNode(self.size_, state, score, num_step_tokens, parent)
        self.size_ += 1
        depth = new_node.get_depth()
        self.depth_nodes[depth].append(new_node)
        self.nodes.append(new_node)
        return
    
    def select_softmax(self, node_list, node_weights, width):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        weights = []
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            weights.append(pair[1])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        for weight in exp_weights:
            num = int(math.ceil(width * weight / sum_exp_weights))
            select_num.append(num)
            width -= num
            sum_exp_weights -= weight
        return nodes, select_num
    
    def select_softmax_with_truncation(self, node_list, node_weights, width):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        weights = []
        truncate_ratio = self.paras["truncate_ratio"]
        keep_num = int(math.ceil(len(sorted_node_weight_pair_list) * truncate_ratio))
        sorted_node_weight_pair_list = sorted_node_weight_pair_list[:keep_num]
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            weights.append(pair[1])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        for weight in exp_weights:
            num = int(math.ceil(width * weight / sum_exp_weights))
            select_num.append(num)
            width -= num
            sum_exp_weights -= weight
        return nodes, select_num



    def select_and_expand(self, depth):
        cand_node_list = []
        cand_node_weights = []
        for node in self.depth_nodes[depth]:
            if node.is_leaf() == True or node.get_cum_tokens() >= self.paras["max_tokens"]:
                self.remaining_width -= 1
            else:
                cand_node_list.append(node)
                cand_node_weights.append(node.get_score())
        if self.remaining_width <= 0 or cand_node_list == []:
            return False
        if self.paras["select_method"] ==  "softmax":
            nodes, widths = self.select_softmax(cand_node_list, cand_node_weights, self.remaining_width)
        if self.paras["select_method"] == "softmax_with_truncate":
            nodes, widths = self.select_softmax_with_truncation(cand_node_list, cand_node_weights, self.remaining_width)
        for expand_node, width in zip(nodes, widths):
            if width >= 1:
                self.expand(expand_node, width)
        return True 
                

@function
def reward_guided_search(s, id, question, ground_truth_answer, paras, reward_host):
    s += question
    tree = Tree(s, paras, reward_host)
    depth = 0
    while True:
        tree.reset_running_list()
        continue_search = tree.select_and_expand(depth)
        if continue_search == False:
            break
        running_list = tree.get_running_list()
        for state, parent in running_list:
            tree.insert(state, parent)
        depth += 1
        if depth >= 25:
            break
    history_list = tree.get_history_list()
    total_tokens = 0
    for state in history_list:
        total_tokens += state.get_meta_info("step")["completion_tokens"]

    all_nodes = tree.get_nodes()
    answers = []
    nodes_info = []
    answer_store_path = paras["store_path"] + f"answer_q{id}.json"
    for node in all_nodes:
        if node.get_parent() is not None:
            parent_id = node.get_parent().get_id()
        else:
            parent_id = None
        nodes_info.append({
            "id": node.get_id(),
            "text": node.get_text(),
            "score": node.get_score(),
            "parent_id": parent_id,
            "depth": node.get_depth(),
            "leaf": node.is_leaf()
        })
        if node.is_leaf():
            step_scores = []
            last_node = node
            while last_node.get_depth() > 0:
                step_scores.append(last_node.get_score())
                last_node = last_node.get_parent()
            step_scores.reverse()
            answers.append({"text":node.get_text(), "step_scores":step_scores})
            

    answer_for_the_question = {"id":id, "question": question, "model_answer":answers, "ground_truth_answer": ground_truth_answer["answer"], "total_tokens":total_tokens}
    json.dump(answer_for_the_question, open(answer_store_path, "w"), indent=4)
    return answer_for_the_question



def search_worker(search_dict, lock, prompts, test_examples, paras, policy_host, reward_host):
    while True:
        q_id = None
        with lock:
            for key in search_dict:
                if search_dict[key] == False:
                    search_dict[key] = True
                    q_id = int(key)
                    break
        if q_id == None:
            break
        state = reward_guided_search.run(id=q_id, question=prompts[q_id], ground_truth_answer=test_examples[q_id], paras=paras, reward_host=RuntimeEndpoint(reward_host), backend=RuntimeEndpoint(policy_host))
        answer_for_the_question = state.ret_value
        return answer_for_the_question



def main(args):
    prompts, test_examples = get_prompts(args)
    with open(args.parameter_path ,'r', encoding='utf-8') as file:
        paras = yaml.safe_load(file)
    input_list_dict = []

    for i, prompt in enumerate(prompts):
        input_list_dict.append({"id":i, "question":prompt, "ground_truth_answer":test_examples[i], "paras":paras, "reward_host":RuntimeEndpoint(args.reward_host)})
    states = reward_guided_search.run_batch(input_list_dict, backend=RuntimeEndpoint(args.policy_host), num_threads=paras["num_threads"], progress_bar=True)

    results = []
    total_gen_tokens = 0
    for s in states:
        answer = s.ret_value
        total_gen_tokens += answer["total_tokens"]
        results.append(answer)

    json.dump(results, open(args.output_path, "w"), indent=4)


    

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--parameter_path', type=str, required=True)
    args_parser.add_argument('--policy_host', type=str)
    args_parser.add_argument('--reward_host', type=str)
    args = args_parser.parse_args()
    main(args)