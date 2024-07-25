import argparse
import json
import os
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import io

data_root_path = "exp_reseults/docci/image_eval"
cache_dir = "pretrained_models"
no_image_path = "No_image_available_forwandb.jpg"
exp_results = [
    "docci/images_eval_output",
    "stable-diffusion-2-1/clip_normal_output",
    "stable-diffusion-2-1/clip_repeattext_output",
    "stable-diffusion-2-1/clip_summary_output",
    "stable-diffusion-v1-5/clip_normal_output",
    "stable-diffusion-v1-5/clip_repeattext_output",
    "stable-diffusion-v1-5/clip_summary_output",
    "stable-diffusion-v1-5/longclip_output"
]
exp_result_root = "exp_results"
vqamodels = ['mpluglarge']
# vqamodels = ['sharegpt4v7b']


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to process ShareGPT4V data")
    parser.add_argument(
        "--wandb_name", type=str, default="docci_exp_show", help="Wandb project name"
    )
    parser.add_argument(
        "--exp_results",
        type=str,
        nargs="+",
        default=exp_results,
        help=f"List of experiment result paths after root path {exp_result_root}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=str,
        choices=[
            "train",
            "test",
            "qual_test",
            "qual_dev",
        ],
        default=["qual_test","qual_dev"],
    )
    return parser.parse_args()


args = parse_arguments()

wandb_name = args.wandb_name
exp_results = args.exp_results
quantitative_root_path = os.path.join(exp_result_root, "quantitative results")

for exp_result in exp_results:
    if os.path.isdir(os.path.join(exp_result_root, exp_result)) is not True:
        raise ValueError(f"Experiment result path {exp_result} does not exist")
print("All experiment result paths exist")

# start a new wandb run to track this script
wandbrun = wandb.init(
    settings=wandb.Settings(disable_git=True),
    # set the wandb project where this run will be logged
    project=wandb_name,
)


docci_id2meta={}
with open("docci_meta_errorfix/docci_metadata_refined.jsonlines", "r") as f:
    for line in f:
        item = json.loads(line)
        key = item.pop("example_id")
        docci_id2meta[key] = item
        
docci_id2desc={}
with open("datasets/docci/docci_descriptions.jsonlines", "r") as f:
    for line in f:
        item = json.loads(line)
        key = item.pop("example_id")
        docci_id2desc[key] = item

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, cache_dir=cache_dir, safety_checker=None
)
tokenizer = pipe.tokenizer


col_exp=[]
for exp_result in exp_results:
    col_exp.append(exp_result+'_img')
    for vqamodel in vqamodels:
        col_exp.append(exp_result+f'_{vqamodel}_avg')
    
columns = (
    ["data_id", "data_type", "prompt"]
    + col_exp + ["score_per_tokenlength"]
)
result_table = wandb.Table(columns=columns)



quanti_dict_results = {}
for vqamodel in vqamodels:
    quanti_dict_results[vqamodel]={}
    for exp_result in exp_results:

        with open(os.path.join(quantitative_root_path,f"evalscore&{exp_result.replace('/','&')}&{vqamodel}_origin.json"),'r') as f:
            quanti_dict_results[vqamodel][exp_result] = json.load(f)
        



for example_id, desc_dict in tqdm(docci_id2desc.items()):
    if desc_dict["split"] in args.splits:
        caption = desc_dict["description"]

        text_input_ids = tokenizer(
            caption,
            padding="longest",
            return_tensors="pt",
        ).input_ids
        front_text = tokenizer.batch_decode(text_input_ids[:, : 77],skip_special_tokens=True)[0]
        backward_text = tokenizer.batch_decode(text_input_ids[:, 77 : -1],skip_special_tokens=True)[0]
        
        front_temp=[]
        for sent_f_id, x in enumerate(front_text.split('.')):
            front_temp.append(f"{sent_f_id+1} : {x.strip()}")
        front_text = '\n'.join(front_temp)
        
        backward_text = [f"{sent_f_id+1+sent_b_id} : {x.strip()}" for sent_b_id, x in enumerate(backward_text.split('.')[:-1])]
        backward_text = '\n'.join(backward_text)
        caption_toshow = f"{front_text}\n{backward_text}"
        
        row_list = [example_id, desc_dict["split"], caption_toshow]
        
        result_lists = []
        sentence_score_perexp = {}            
        for exp_result in exp_results:
            split = desc_dict["split"]
            exp_result_path = os.path.join(exp_result_root, exp_result)
            image_path = os.path.join(exp_result_path, split, f"{example_id}.jpg")
            if os.path.isfile(image_path) is not True:
                image_path = no_image_path
            result_lists.append(wandb.Image(image_path))
            
            sentence_score_perexp[exp_result]={}
            for vqamodel in vqamodels:
                avg_score_img=[]
                for sent_id, sent_info in docci_id2meta[example_id]['dsg']['question'].items():
                    if len(sent_info['question'])>0:
                        score_key = f"{example_id}&{sent_id}"
                        score_avg_dictqa = quanti_dict_results[vqamodel][exp_result][score_key]
                        avg_score_sentence = score_avg_dictqa[0]
                        avg_score_img.append(avg_score_sentence)
                        
                        sentence_score_perexp[exp_result][sent_id] = avg_score_sentence                        
                    else:
                        sentence_score_perexp[exp_result][sent_id] = 0 
                        
                avg_score_img = np.mean(avg_score_img)
                result_lists.append(avg_score_img)
        
        plt.figure(figsize=(12, 6))
        bar_width = 0.1  # Width of the bars
        index = np.arange(len(list(sentence_score_perexp.values())[0]))  # Sentence numbers
        
        sd21_colors = ['#0000FF', '#4D4DFF', '#9999FF', '#CCCCFF', '#E6E6FF']  # Blue to light blue
        sd15_colors = ['#008000', '#4CAF50', '#80FF80', '#B3FFB3', '#E6FFE6']  # Green to light green
        
        color_map = []
        for exp_result in sentence_score_perexp.keys():
            exp_result = exp_result.replace("stable-diffusion-2-1","SD2.1")
            exp_result = exp_result.replace("stable-diffusion-v1-5","SD1.5")
            if "SD2.1" in exp_result:
                color_map.append(sd21_colors.pop(0))
            elif "SD1.5" in exp_result:
                color_map.append(sd15_colors.pop(0))
            else:
                color_map.append("#000000")  # Default color if neither SD2.1 nor SD1.5

        for i, (exp_result, scores) in enumerate(sentence_score_perexp.items()):
            sentences = list(scores.keys())
            avg_scores = list(scores.values())
            plt.bar(index + i * bar_width, avg_scores, bar_width, label=exp_result, color=color_map[i])
        
        plt.xlabel("Sentence Number")
        plt.ylabel("Score")
        plt.title("Sentence Scores per Experiment Result")
        plt.xticks(index + bar_width * (len(sentence_score_perexp) - 1) / 2, sentences)
        plt.legend()
        plt.grid(True)
        
        # Save the plot as an image
        plot_image_path = f"sentence_scores.jpg"
        plt.savefig(plot_image_path)
        plt.close()

        # Open the saved image using PIL and append to result lists
        plot_image = Image.open(plot_image_path)
        result_lists.append(wandb.Image(plot_image))
        
        # plus data for all columns
        row_list = row_list + result_lists
        result_table.add_data(*row_list)
wandbrun.log({"table_key":result_table})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()







        # token2score_img = {}
        # for i in range(2, list(docci_id2meta[example_id]['dsg']['question'].items())[-1][1]['end_tokenlen']+1):
        #     token2score_img[i] = []
            
        # for exp_result in exp_results:
        #     token2score_img[exp_result] = {}
        #     for sent_id, sent_info in docci_id2meta[example_id]['dsg']['question'].items():
        #         if len(sent_info['question'])>0:
        #             score_key = f"{example_id}&{sent_id}"
        #             avg_score_sentence = quanti_dict_results[vqamodel][exp_result][score_key][0]
        #             for i in token2score_img.keys():
        #                 if sent_info['start_tokenlen']<=i and i<=sent_info['end_tokenlen']:
        #                     token2score_img[exp_result][i]=avg_score_sentence
        # for i in token2score_img.keys():
        #     if sent_info['start_tokenlen']<=i and i<=sent_info['end_tokenlen']:
        #         token2score[exp_result][i].append(avg_score_sentence)