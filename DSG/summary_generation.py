import string
from typing import Callable
from llama_utils import groq_setup, groq_completion
import os
import json
from tqdm import tqdm
import time


_PROMPT_TEMPLATE = string.Template("""Generate summary for the following scene.
Generate the summary within 80 words.
No bulletpoints or explanations needed.
Just output the summary text.

Scene: $input_text

Summary: """.strip())

base_dir = "/home/compu/JinProjects/jinprojects/SELMA/datasets/ShareGPT4V/data"
sampled_json_path = os.path.join(
    base_dir,
    "sharegpt4v/FixedDict_max1ksampled_sharegpt4v_instruct_gpt4-vision_cap100k.json",
)
output_json_path = os.path.join(
    base_dir,
    "sharegpt4v/wsww_summary_sharegpt4v_instruct_gpt4-vision_cap100k_frontback.json",
)
error_json_path = os.path.join(
    base_dir,
    "sharegpt4v/wsww_summary_sharegpt4v_instruct_gpt4-vision_cap100k_frontback_Errorlist.json",
)


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


with open(sampled_json_path, "r") as f:
    json_data = json.load(f)

for folder in ['wikiart', 'share_textvqa', 'web-celebrity', 'web-landmark','coco', 'sam', 'llava']:
    datadict = json_data[folder]
    for idx in tqdm(datadict):
        iddict = datadict[idx]
        gpt4vcaption = iddict["total"]["caption"]
        error_list = {}
        id = idx
        imagepath = iddict["imagepath"]
        output_dump_list = {
            idx: {
                "imagepath": imagepath,
                "caption": gpt4vcaption,
            }
        }

        try:
            time.sleep(6)
            summary_prompt = generate_summary_prompt(generate_fn=groq_completion, input_text=gpt4vcaption)
        except Exception as e:
            error_list[id] = {}
            error_list[id]["error_msg"] = gpt4vcaption+"\n"+str(e)
            with open(error_json_path, "a") as f:
                json.dump(error_list, f, indent=4)
                f.write(", \n")
            continue
        print("\n")
        output_dump_list[idx]["summary"] = summary_prompt
        with open(output_json_path, "a") as f:
            json.dump(output_dump_list, f, indent=4)
            f.write(", \n")
        print(f"Successfully generated summary for image {id}.\n\n")
