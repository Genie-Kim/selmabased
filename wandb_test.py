import wandb
import random
import re

# start a new wandb run to track this script
wandb.init(
    settings=wandb.Settings(disable_git=True),
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()





# import json


# def clean_thelimbs(output_str):
#     if 'outputs:' in output_str:
#         start_index = output_str.index('outputs:')
#         output_str = output_str[start_index+len('outputs:'):]
#         output_str = output_str.strip()
        
#     if 'output:' in output_str:
#         start_index = output_str.index('output:')
#         output_str = output_str[start_index+len('output:'):]
#         output_str = output_str.strip()
        
#     if '\n\n' in output_str:
#         end_index = output_str.rfind('\n\n')
#         output_str = output_str[:end_index]
#         output_str = output_str.strip()
#     return output_str




# def find_elements(text):
#     # Define the regex pattern strictly for numbers between \n and | with possible spaces
#     pattern = r'\s*(\d+)\s*\|(.+)\n'
    
#     # Find all matches
#     matches = re.findall(pattern, text)
    
#     # Return the list of numbers found (as strings, could convert to int if needed)
#     return matches


# def find_last_number_between_newline_and_pipe(text):
#     # Define the regex pattern strictly for numbers between \n and | with possible spaces
#     pattern = r'\n\s*(\d+)\s*\|'
    
#     # Find all matches
#     matches = re.findall(pattern, text)
    
#     # Return the last number found, or None if no match
#     return matches[-1] if matches else None


# def question_num_check(id2tup, id2ques, id2dep):
#     id2tup_num=find_last_number_between_newline_and_pipe(id2tup)
#     id2ques_num=find_last_number_between_newline_and_pipe(id2ques)
#     id2dep_num=find_last_number_between_newline_and_pipe(id2dep)
#     if len(id2tup_num) != len(id2ques_num) or len(id2ques_num) != len(id2dep_num):
#         return False
#     return True

# with open("/home/compu/JinProjects/jinprojects/SELMA/datasets/ShareGPT4V/data/sharegpt4v/DSG_generated_sharegpt4v_instruct_gpt4-vision_cap100k_Errorlist.json", 'r') as f:
#     dataset = json.load(f)
# count = 0

# for item in dataset:
#     for key, value in item.items():
#         id2tuple_origin = value["id2tuple"]
#         id2question_origin = value["id2question"]
#         id2dependency_origin = value["id2dependency"]
        
#         find_last_number_between_newline_and_pipe(id2tuple_origin)
#         find_last_number_between_newline_and_pipe(id2question_origin)
#         find_last_number_between_newline_and_pipe(id2dependency_origin)
        
        
#         id2tuple=clean_thelimbs(id2tuple_origin).split('\n')
#         id2question=clean_thelimbs(id2question_origin).split('\n')
#         id2dependency=clean_thelimbs(id2dependency_origin).split('\n')
#         if len(id2tuple) != len(id2question) or len(id2question) != len(id2dependency):
#             print(key)
#             print(len(id2tuple), len(id2question), len(id2dependency))
#             print(id2tuple_origin)
#             print(id2question_origin)
#             print(id2dependency_origin)
#             count+=1
            
# print(count)    
        
        