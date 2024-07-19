import string
from typing import Callable
from llama_utils import groq_setup, groq_completion
import os
import json
from tqdm import tqdm
import time
from pprint import pprint

from diffusers import StableDiffusionPipeline
import torch

_PROMPT_TEMPLATE = string.Template("""Generate summary for the following scene.
Generate the summary within 80 words.
No bulletpoints or explanations needed.
Just output the summary text.

Scene: $input_text

Summary: """.strip())

base_dir = "/home/compu/JinProjects/jinprojects/SELMA/datasets/docci/"
docci_desc_jsonl = os.path.join(
    base_dir,
    "docci_descriptions.jsonlines",
)
docci_meta_jsonl = os.path.join(
    base_dir,
    "docci_metadata.jsonlines",
)
output_json_path = os.path.join(
    base_dir,
    "summary_docci.jsonlines",
)
error_json_path = os.path.join(
    base_dir,
    "summary_docci_error.jsonlines",
)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    cache_dir="pretrained_models",
    safety_checker=None,
)
tokenizer = pipe.tokenizer


def get_token_length(prompt: str):
    text_input_ids = tokenizer(
        prompt,
        padding="longest",
        return_tensors="pt",
    ).input_ids
    return len(text_input_ids[0])


def generate_summary_prompt(
        generate_fn: Callable[[str], str],
        input_text: str,
    ) -> str:
        
    input_text = input_text.replace("\n\n","\n")
    input_text = input_text.replace("**","")
    
    prompt = _PROMPT_TEMPLATE.substitute(
        input_text=input_text,
    )
    
    output = generate_fn(prompt)
    print(prompt)
    print(output)
    
    return output


groq_setup()
print(
    groq_completion(
        "hello, how are you doing?",
    )
)
docci_id2info={}
with open(docci_desc_jsonl, "r") as f:
    for line in f:
        item = json.loads(line)
        key = item.pop("example_id")
        docci_id2info[key] = item
        
docci_id2meta={}
with open(docci_meta_jsonl, "r") as f:
    for line in f:
        item = json.loads(line)
        key = item.pop("example_id")
        docci_id2meta[key] = item
        
for example_id, values in tqdm(docci_id2info.items()):
    print("\n\n##################################################")
    if 'train' in example_id:
        prompt = values['description']
        values['example_id'] = example_id
        
        token_length = get_token_length(prompt)
        values['token_length'] = token_length
        
        print("\n\nimageid : ",example_id)
        print("token_length",token_length)
        if token_length > 77:
            try:
                time.sleep(6)
                summary_prompt = generate_summary_prompt(generate_fn=groq_completion, input_text=prompt)
            except Exception as e:
                error_dict={
                    'example_id' : example_id,
                    'description' : prompt,
                    'error_msg' : str(e)
                }
                with open(error_json_path, "a") as f:
                    json.dump(error_dict, f, indent=4)
                    f.write(", \n")
                print("\n\n!!!!!!!!!!ERROR!!!!!!!!!!")
                pprint(error_dict)
                continue
            
            values['long_bool']=True
            values["summary"] = summary_prompt
            values["summary_token_length"] = get_token_length(summary_prompt)
            
        else:
            values['long_bool']=False
            print("Short prompt :", prompt)
                
        with open(output_json_path, "a") as f:
            f.write(json.dumps(values)+"\n")
        print(f"\nSuccessfully generated summary for image {example_id}.\n\n")

# sharegpt4v
# import string
# from typing import Callable
# from llama_utils import groq_setup, groq_completion
# import os
# import json
# from tqdm import tqdm
# import time


# _PROMPT_TEMPLATE = string.Template("""Generate summary for the following scene.
# Generate the summary within 80 words.
# No bulletpoints or explanations needed.
# Just output the summary text.

# Scene: $input_text

# Summary: """.strip())

# base_dir = "/home/compu/JinProjects/jinprojects/SELMA/datasets/ShareGPT4V/data"
# sampled_json_path = os.path.join(
#     base_dir,
#     "sharegpt4v/FixedDict_max1ksampled_sharegpt4v_instruct_gpt4-vision_cap100k.json",
# )
# output_json_path = os.path.join(
#     base_dir,
#     "sharegpt4v/wsww_summary_sharegpt4v_instruct_gpt4-vision_cap100k_frontback.json",
# )
# error_json_path = os.path.join(
#     base_dir,
#     "sharegpt4v/wsww_summary_sharegpt4v_instruct_gpt4-vision_cap100k_frontback_Errorlist.json",
# )


# def generate_summary_prompt(
#         generate_fn: Callable[[str], str],
#         input_text: str,
#     ) -> str:
        
#     input_text = input_text.replace("\n\n","\n")
#     input_text = input_text.replace("**","")
    
#     prompt = _PROMPT_TEMPLATE.substitute(
#         input_text=input_text,
#     )
    
#     output = generate_fn(prompt)
#     print(prompt)
#     print(output)
    
#     return output


# groq_setup()
# print(
#     groq_completion(
#         "hello, how are you doing?",
#     )
# )


# with open(sampled_json_path, "r") as f:
#     json_data = json.load(f)


# for folder in ['wikiart', 'share_textvqa', 'web-celebrity', 'web-landmark','coco', 'sam', 'llava']:
#     datadict = json_data[folder]
#     for idx in tqdm(datadict):
#         if idx in ['sa_27924','Delphi2','Machu_Picchu']:
#             iddict = datadict[idx]
#             gpt4vcaption = iddict["total"]["caption"]
#             error_list = {}
#             id = idx
#             imagepath = iddict["imagepath"]
#             output_dump_list = {
#                 idx: {
#                     "imagepath": imagepath,
#                     "caption": gpt4vcaption,
#                 }
#             }

#             try:
#                 time.sleep(6)
#                 summary_prompt = generate_summary_prompt(generate_fn=groq_completion, input_text=gpt4vcaption)
#             except Exception as e:
#                 error_list[id] = {}
#                 error_list[id]["error_msg"] = gpt4vcaption+"\n"+str(e)
#                 with open(error_json_path, "a") as f:
#                     json.dump(error_list, f, indent=4)
#                     f.write(", \n")
#                 continue
#             print("\n")
#             output_dump_list[idx]["summary"] = summary_prompt
#             with open(output_json_path, "a") as f:
#                 json.dump(output_dump_list, f, indent=4)
#                 f.write(", \n")
#             print(f"Successfully generated summary for image {id}.\n\n")
