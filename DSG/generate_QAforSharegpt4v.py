import openai
from openai_utils import openai_setup, openai_completion
from llama_utils import groq_setup, groq_completion
from query_utils import generate_dsg
from parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
import json
import os
import json
import random
import time
import pandas as pd
import re
# =====================================
# Task 1: Random sample 500 images per dataset from jsonfile.
# =====================================
# base_dir = "/home/compu/JinProjects/jinprojects/SELMA/datasets/ShareGPT4V/data"
# sharecaptioner_json_path = "sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json"
# sharecaptioner_json_path = os.path.join(base_dir, sharecaptioner_json_path)

# gpt4v_path = "sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json"
# gpt4v_path = os.path.join(base_dir, gpt4v_path)


# with open(gpt4v_path, 'r') as f:
#     gpt4v_json = json.load(f)
    
# with open(sharecaptioner_json_path, 'r') as f:
#     sharecaptioner_json = json.load(f)


# sharecaptioner_dict = {}
# for item in sharecaptioner_json:
#     if item['id'] in sharecaptioner_dict:
#         print(f"Duplicate ID found: {item['id']}")
#     else:
#         sharecaptioner_dict[item['id']] = {
#             'image' : item['image'],
#             'conversations' : item['conversations'],
#             }


# def find_caption_from_json(jsondata, id):
#     if id in jsondata:
#         return jsondata[id]['conversations'][1]['value']
#     else:
#         return None



# # Dictionary to store image paths grouped by their folders
# image_groups = {}
# # Group images by their folders
# for item in gpt4v_json:
#     image_path = item['image']
#     folder = image_path.split('/')[0]
#     if folder in ['coco','sam','llava']:
#         caption = find_caption_from_json(sharecaptioner_dict, item['id'])
#         if caption is None:
#             print(f"Caption not found for image {item['id']}. Skipping...") 
#         else:
#             template = {'from':'sharecaptioner','value':caption}
#             item['conversations'].append(template)
#     if folder not in image_groups:
#         image_groups[folder] = []
#     image_groups[folder].append(item)

# merged_file_path = 'mergedlist_sharegpt4v_instruct_gpt4-vision_cap100k.json'
# with open(merged_file_path, 'w') as f:
#     json.dump(image_groups, f, indent=4)
    

# # Dictionary to store sampled image paths
# sampled_images = {}

# # Function to sample images
# def sample_images(item_list, sample_size=500):
#     return random.sample(item_list, min(sample_size, len(item_list)))

# # Sample images from each group and store the paths
# for folder, items in image_groups.items():
#     sampled_images[folder] = sample_images(items)
    
# # Optionally, write the sampled image paths to a file
# output_file_path = '500sampledlist_share-captioner_coco_lcs_sam_1246k_1107.json'
# with open(output_file_path, 'w') as f:
#     json.dump(sampled_images, f, indent=4)

# print(f"Sampled image paths have been written to {output_file_path}.")


# # =====================================
# # Task 2: Question generation
# # =====================================

# # Reading API KEY from _OAI_KEY.txt
# base_dir = "/home/compu/JinProjects/jinprojects/SELMA/datasets/ShareGPT4V/data"
# sampled_json_path = 'sharegpt4v/500sampledlist_share-captioner_coco_lcs_sam_1246k_1107.json'
# output_json_path='datasets/ShareGPT4V/data/sharegpt4v/DSG_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json'
# error_json_path = 'datasets/ShareGPT4V/data/sharegpt4v/DSG_generated_sharegpt4v_instruct_gpt4-vision_cap100k_Errorlist.json'
# sampled_json_path = os.path.join(base_dir, sampled_json_path)

# groq_setup()
# print(groq_completion(
#     'hello, how are you doing?',
# ))

# openai_setup()
# print(openai_completion(
#     'hello, how are you doing?',
# ))

# with open(sampled_json_path, 'r') as f:
#     json_data = json.load(f)

# keywordpattern  = r'\((.*?)\)'

# for folder, datalist in json_data.items():
#     for data in datalist:
#         error_list = {}
#         output_dump_list = {}
        
        
#         id = data['id']
#         imagepath = data['image']
#         INPUT_TEXT_PROMPT = data['conversations'][1]['value'].replace('\n\n','\n')
#         id2prompts = {
#             'custom_0': {
#                 'input': INPUT_TEXT_PROMPT,
#             }
#         }

#         id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
#             id2prompts,
#             # you can change this method with any method that takes prompt as input and outputs LLM generation result.
#             generate_fn=groq_completion
#         )
#         try:
#             qid2tuple = parse_tuple_output(id2tuple_outputs['custom_0']['output'])
#             qid2dependency = parse_dependency_output(id2dependency_outputs['custom_0']['output'])
#             qid2question = parse_question_output(id2question_outputs['custom_0']['output'])
            
#             question_number = len(qid2question)
#             qid2tuple = {k: v for k, v in qid2tuple.items() if k <= question_number}
#             qid2dependency = {k: v for k, v in qid2dependency.items() if k <= question_number}
                            
#             tupdict = {}
#             keyworddict = {}
#             output_dump_list[id]={
#             'gpt4vcaption': INPUT_TEXT_PROMPT,
#             'imagepath': imagepath,
#             'question_list':[]
#             }
#             for tup in id2tuple_outputs['custom_0']['output'].strip().split('\n'):
#                 id_temp = int(tup.split('|')[0].strip())
#                 tupdict[id_temp]=tup.split('|')[1].strip()
#                 keyworddict[id_temp] = re.search(keywordpattern, tupdict[id_temp]).group(1)
            
#             for i in range(1, question_number+1):
#                 keywords = keyworddict[i]
#                 proposition_id = str(i)
#                 dependency = ','.join([str(idx) for idx  in qid2dependency[i]])
#                 category_broad = qid2tuple[i].split('-')[0].strip()
#                 category_detailed = qid2tuple[i].split('-')[1].strip()
#                 tuples = tupdict[i]
#                 question_natural_language = qid2question[i]
#                 output_dump_list[id]['question_list'].append({
#                     'keywords': keywords,
#                     'proposition_id': proposition_id,
#                     'dependency': dependency,
#                     'category_broad': category_broad,
#                     'category_detailed': category_detailed,  
#                     'tuples': tuples,
#                     'question_natural_language': question_natural_language
#                 })
#             with open(output_json_path , 'a') as f:
#                 json.dump(output_dump_list, f, indent=4)
#                 f.write(', \n')
#             print(f"Successfully generated questions for image {id}.\n\n\n")

#         except Exception as e:
#             error_list[id] = {}
#             error_list[id]['id2tuple'] = id2tuple_outputs['custom_0']['output']
#             error_list[id]['id2question'] = id2question_outputs['custom_0']['output']
#             error_list[id]['id2dependency'] = id2dependency_outputs['custom_0']['output']
#             error_list[id]['error_msg'] = str(e)
#             with open(error_json_path, 'a') as f:
#                 json.dump(error_list, f, indent=4)
#                 f.write(', \n')
                
            
#         time.sleep(6)
# After the json file generated, you should add [,] to the beginning and end of the file.


# =====================================
# Task 3: Check the generated questions file.
# =====================================

def isEngNum(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
output_json_path='datasets/ShareGPT4V/data/sharegpt4v/DSG_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json'
error_json_path = 'datasets/ShareGPT4V/data/sharegpt4v/DSG_generated_sharegpt4v_instruct_gpt4-vision_cap100k_Errorlist.json'
sampled_json_path = 'datasets/ShareGPT4V/data/sharegpt4v/500sampledlist_share-captioner_coco_lcs_sam_1246k_1107.json'

with open(output_json_path,'r') as f:
    DSG_sharegpt4v_data = json.load(f)
    
with open(error_json_path,'r') as f:
    DSG_sharegpt4v_Error_data = json.load(f)

with open(sampled_json_path,'r') as f:
    origin_data = json.load(f)
    
print(f"Successfully generated questions for {len(DSG_sharegpt4v_data)} images.")
print(f"Failed to generate questions for {len(DSG_sharegpt4v_Error_data)} images.")

origin_data_dict = {}
for folder,dictlist in origin_data.items():
    for dict1 in dictlist:
        tempdict = dict1.copy()
        id = tempdict.pop('id')
        value = tempdict
        if folder not in origin_data_dict:
            origin_data_dict[folder] = {}
        origin_data_dict[folder][id] = value


print([(k,len(v)) for k,v in origin_data.items()])
print([(k,len(v)) for k,v in origin_data_dict.items()])


idlist = []
for k,v in origin_data.items():
    for item in v:
        idlist.append(item['id'])
# count the histogram of id in idlist
idhist = {x:idlist.count(x) for x in idlist}
# print all the id with count higher than 1
print({k:v for k,v in idhist.items() if v>1})
dup_ids = {k:v for k,v in idhist.items() if v>1}.keys()


# change the DSG_sharegpt4v_data list to dictionary with folder name as key and id as subkey
DSG_sharegpt4v_dict = {}
for item in DSG_sharegpt4v_data:
    for id,row in item.items():
        if id not in dup_ids: # duplicate id from the original data should be removed
            if isEngNum(row['gpt4vcaption']):
                folder = row['imagepath'].split('/')[0]
                if folder not in DSG_sharegpt4v_dict:
                    DSG_sharegpt4v_dict[folder] = {}
                DSG_sharegpt4v_dict[folder][id] = row


print([len(v) for k,v in DSG_sharegpt4v_dict.items()])

with open('datasets/ShareGPT4V/data/sharegpt4v/Dict_DSG_nondupid_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json','w') as f:
    json.dump(DSG_sharegpt4v_dict, f, indent=4)



# change the DSG_sharegpt4v_Error_data list to dictionary with folder name as key and id as subkey
error_dict = {}
for item in DSG_sharegpt4v_Error_data:
    for id,row in item.items():
        for k,v in origin_data_dict.items():
            if id in v:
                folder = k
                if folder not in error_dict:
                    error_dict[folder] = {}
                error_dict[folder][id] = row
        
print([len(v) for k,v in error_dict.items()])
        
