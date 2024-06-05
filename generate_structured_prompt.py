import openai
from DSG.openai_utils import openai_setup, openai_completion
from DSG.llama_utils import groq_setup, groq_completion
from DSG.query_utils import generate_dsg
from DSG.parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
import json
import os
import json
import random
import time
import pandas as pd
import re
from collections import deque

def append_to_dict_list(d, key, value):
    if key not in d:
        d[key] = []
    d[key].append(value)

def generate_structured_prompt(prompt, dependency_tree):
    # TODO : generate structured prompt from new prompts.
    for k,v in dependency_tree.items():
        if k == 'global':
            global_lists = v
        elif k == 'two_relations':
            two_relation_lists = v
        elif k == 'etc':
            etc_lists = v
        else:
            
            pass
    pass

def find_path_to_root(parents, question_list):
    queue = deque(parents)
    while queue:
        current = queue.popleft()
        for next_node in question_list:
            if next_node['proposition_id'] == current:
                ys_dep = []
                for p in next_node['dependency'].split(','):
                    p = p.strip()
                    if p != '0': # 0을 제외한 나머지를 temp_dep에 추가
                        ys_dep.append(p)
                        
                if len(ys_dep)==0 or next_node['category_broad'].strip()=='global':
                    # root 바로 아래 node가 맞으면, 그냥 return
                    return current
                else:
                    # root 바로 아래 node가 아니면, queue에 추가
                    for x in ys_dep:
                        queue.append(x)
    return None
    


def structured_prompt_from_dsgdata(id, data):
    
    # get the tuple and dependency from the data
    gpt4vcaption = row['gpt4vcaption']
    question_list = row['question_list']
    
    depen_tree = {}
    for quesinfo in question_list:
        temp_dep = []
        for x in quesinfo['dependency'].split(','):
            x = x.strip()
            if x != '0': # 0을 제외한 나머지를 temp_dep에 추가
                temp_dep.append(x)
        
        category_broad = quesinfo['category_broad'].strip()
        # if (category_broad=='relation' or category_broad=='action') and len(temp_dep)>1:
        if (category_broad=='relation') and len(temp_dep)>1:
            # relation or action에 대한 tuple인데, dependency가 0을 제외하고 두개를 가지고 있는 경우.
            append_to_dict_list(depen_tree,'two_relations',quesinfo)
        
        elif len(temp_dep)==0 or category_broad=='global':
            # dependency가 0 이거나 global인 경우.
            append_to_dict_list(depen_tree,'global',quesinfo)
        else:
            # 어딘가에 종속되어야 하는 경우.
            key_parent_id = find_path_to_root(temp_dep, question_list)
            if key_parent_id is not None:
                append_to_dict_list(depen_tree,key_parent_id,quesinfo)
            else:
                append_to_dict_list(depen_tree,'etc',quesinfo)

    generate_structured_prompt(gpt4vcaption, depen_tree)
    
    return structured_prompt_dict
    


# =====================================
# Task 1: load the dependency graph from DSG like sharegpt4v prompts.
# =====================================

dsg_sharegpt4v_path = 'testing_dsgjson.json'
with open(dsg_sharegpt4v_path,'r') as f:
    dsg_sharegpt4v = json.load(f)


# change the list of dictionaries to dictionary with id key
dsg_sharegpt4v_dict = {}
for item in dsg_sharegpt4v:
    for id,row in item.items():
        dsg_sharegpt4v_dict[id] = row

test_id = 'sa_16091'
structured_prompt_dict = structured_prompt_from_dsgdata(test_id,dsg_sharegpt4v_dict[test_id])



print(1)

actionlist = []
for item in dsg_sharegpt4v:
    for id,row in item.items():
        for que in row['question_list']:
            if 'action' == que['category_broad']:
                actionlist.append(que)