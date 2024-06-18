import argparse
import json
import os
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import wandb


# def get_result_paths(exp_result_paths, id):
#     result_paths = []
#     for exp_result_path in exp_result_paths:
#         count = 0
#         for result_path in os.listdir(exp_result_path):
#             if os.path.splitext(result_path)[0] == id:
#                 result_paths.append(os.path.join(exp_result_path, result_path))
#                 count += 1
#                 break
#         if count == 0:
#             result_paths.append(None)
#     return result_paths


# def get_origin_info(origin_json, id):
#     for folder, value in origin_json.items():
#         if id in value:
#             return (folder, value[id])
#     return None


origin_json_path = "datasets/ShareGPT4V/data/sharegpt4v/Dict_DSG_nondupid_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json"
data_root_path = "/hddsdb/sharegpt/ShareGPT4V/data"
cache_dir = "pretrained_models"
no_image_path = "No_image_available_forwandb.jpg"
exp_results = [
    "stable-diffusion-v1-5/clip_repeat_textenc_False",
    "stable-diffusion-v1-5/clip_repeat_textenc_True",
    "stable-diffusion-v1-5/longclip_repeat_textenc_False",
    "stable-diffusion-2/clip_repeat_textenc_False",
    "stable-diffusion-2/clip_repeat_textenc_True",
]
exp_result_root = "exp_results"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to process ShareGPT4V data")
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model ID to be used",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=cache_dir,
        help="Directory to cache pretrained models",
    )
    parser.add_argument(
        "--wandb_name", type=str, default="exp_results_show", help="Wandb project name"
    )
    parser.add_argument(
        "--exp_results",
        type=str,
        nargs="+",
        default=exp_results,
        help=f"List of experiment result paths after root path {exp_result_root}",
    )
    return parser.parse_args()


args = parse_arguments()

model_id = args.model_id
cache_dir = args.cache_dir
wandb_name = args.wandb_name
exp_results = args.exp_results
quantitative_root_path = os.path.join(exp_result_root, "quantitative results")



# start a new wandb run to track this script
wandbrun = wandb.init(
    settings=wandb.Settings(disable_git=True),
    # set the wandb project where this run will be logged
    project=wandb_name,
)

with open(origin_json_path, "r") as f:
    origin_json = json.load(f)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, cache_dir=cache_dir, safety_checker=None
)
tokenizer = pipe.tokenizer


col_exp=[]
for exp_result in exp_results:
    col_exp.append(exp_result+'_img')
    col_exp.append(exp_result+'_mPlug_avg')
    col_exp.append(exp_result+'_QA')
    
columns = (
    ["data_id", "data_type", "image_path", "prompt", "clip_removed_text", "longclip_removed_text","original_image"]
    + col_exp
)
result_table = wandb.Table(columns=columns)



quanti_dict_results = {}
for exp_result in exp_results:
    exp_result_path = os.path.join(exp_result_root, exp_result)

    with open(os.path.join(quantitative_root_path,f"evalscore_{exp_result.replace('/','_')}.json"),'r') as f:
        quanti_dict_results[exp_result] = json.load(f)
        



for folder, value in origin_json.items():
    for idx, value_dict in tqdm(value.items()):
        origin_imgpath = os.path.join(data_root_path, value_dict["imagepath"])
        filename = os.path.basename(origin_imgpath)

        gpt4vcaption = value_dict["gpt4vcaption"]
        question_dictlist = value_dict["question_list"]
        question_list = [item["question_natural_language"] for item in question_dictlist]
        question_list_str = '\n'.join(question_list)

        text_input_ids = tokenizer(
            gpt4vcaption,
            padding="longest",
            return_tensors="pt",
        ).input_ids
        clip_removed_text = tokenizer.batch_decode(text_input_ids[:, 77 - 1 : -1])[0]
        longclip_removed_text = tokenizer.batch_decode(text_input_ids[:, 248 - 1 : -1])[0]


        origin_img = wandb.Image(origin_imgpath)
        row_list = [idx, folder, filename, gpt4vcaption, clip_removed_text, longclip_removed_text, origin_img]
        
        result_lists = []
        for exp_result in exp_results:
            exp_result_path = os.path.join(exp_result_root, exp_result)
            for result_path in os.listdir(exp_result_path):
                count = 0
                if os.path.splitext(result_path)[0] == idx:
                    count+=1
                    break
            if count==0:
                result_path = no_image_path

            score_avg_dictqa = quanti_dict_results[exp_result][folder][idx]
            avg_score =score_avg_dictqa[0]
            qa_dict =score_avg_dictqa[1]
            
            
            QA_list = []
            for qid,question in enumerate(question_list):
                ans=qa_dict[str(qid+1)]
                if ans==1:
                    ans='yes'
                else:
                    ans='no'
                QA_list.append(f"{question} : {ans}")
            
            
            result_lists.append(wandb.Image(os.path.join(exp_result_path,result_path)))
            result_lists.append(avg_score)
            result_lists.append('\n'.join(QA_list))
        # plus data for all columns
        row_list = row_list + result_lists
        result_table.add_data(*row_list)
wandbrun.log({"table_key":result_table})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()







