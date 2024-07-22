# parseargs.py
import json
from pprint import pprint
datapath = "datasets/docci/docci_metadata.jsonlines"

docci_id2meta={}
with open(datapath, "r") as f:
    for line in f:
        item = json.loads(line)
        key = item.pop("example_id")
        docci_id2meta[key] = item
            
# fix DSG dependency error samples
target_ids = ["test_00572","test_02388","test_02760"]
target_sent_ids = ["3","9","10"]
for example_id, sentence_id in zip(target_ids, target_sent_ids):
    with open(f"docci_meta_errorfix/{example_id}&{sentence_id}.jsonl", "r") as f:
        fixed_sent_dicts = [json.loads(line) for line in f]

    sent_dict = docci_id2meta[example_id]['dsg']['question'][sentence_id]

    sent_dict['question']={}
    sent_dict['dependency']={}
    sent_dict['tuple']={}
    for item in fixed_sent_dicts:
        qid = item['proposition_id']
        category_broad = item['category_broad']
        category_detailed = item['category_detailed']
        keywords = item['keywords']
        tuple1=f"{category_broad} - {category_detailed} ({keywords})"
        question = item['question_natural_language']
        dependency = item['dependency']
        
        sent_dict['dependency'][qid]=dependency
        sent_dict['tuple'][qid]=tuple1
        sent_dict['question'][qid]=question
    docci_id2meta[example_id]['dsg']['question'][sentence_id] = sent_dict


# refine docci meta jsonline file.

import re
from diffusers import StableDiffusionPipeline
import torch

docci_id2desc={}
with open("datasets/docci/docci_descriptions.jsonlines", "r") as f:
    for line in f:
        item = json.loads(line)
        key = item.pop("example_id")
        docci_id2desc[key] = item

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    cache_dir="pretrained_models",
    safety_checker=None,
)
tokenizer = pipe.tokenizer

for example_id, values in docci_id2meta.items():
    if 'dsg' in values:
        start_token_length = 2 # because sos
        for sent_id, sent_dict in values['dsg']['question'].items():
            sent_dict['cat_broad'] = {}
            sent_dict['cat_detail'] = {}
            sent_dict['keyword'] = {}

            text_input_ids = tokenizer(
                sent_dict['prompt'],
                padding="longest",
                return_tensors="pt",
            ).input_ids
            end_token_length = start_token_length + len(tokenizer.convert_ids_to_tokens(text_input_ids[0],skip_special_tokens=True))
            sent_dict['start_tokenlen']=start_token_length
            sent_dict['end_tokenlen']=end_token_length-1
            start_token_length = end_token_length
            if len(sent_dict['question'])>0:
                for qid, q in sent_dict['question'].items():
                    tuple_str = sent_dict['tuple'][qid]
                    dependency_ids = sent_dict['dependency'][qid].strip().split(',')
                    dependency_ids = [dep_id.strip() for dep_id in dependency_ids]

                    matches = re.search(r"(.*?)\-(.*?)\((.*?)\)", tuple_str)
                    keyword = matches.group(3).strip()
                    category_broad = matches.group(1).strip()
                    category_detailed = matches.group(2).strip()
                    sent_dict['cat_broad'][qid] = category_broad
                    sent_dict['cat_detail'][qid] = category_detailed
                    sent_dict['keyword'][qid] = keyword
                    sent_dict['dependency'][qid] = dependency_ids
        print(end_token_length)


with open("docci_meta_errorfix/docci_metadata_refined.jsonlines", "w") as f:
    for example_id, values in docci_id2meta.items():
        values['example_id']= example_id
        f.write(json.dumps(values))
        f.write("\n")