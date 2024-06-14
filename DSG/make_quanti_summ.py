import csv
import os
import json
from tqdm import tqdm
import numpy as np
import argparse

exp_result_path ="exp_results"
exp_results = [
    "stable-diffusion-v1-5/clip_repeat_textenc_False",
    "stable-diffusion-v1-5/clip_repeat_textenc_True",
    "stable-diffusion-v1-5/longclip_repeat_textenc_False",
    "stable-diffusion-2/clip_repeat_textenc_False",
    "stable-diffusion-2/clip_repeat_textenc_True",
]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_results",
    type=str,
    nargs="+",
    default=exp_results,
    help=f"List of experiment result paths after root path {exp_result_path}",
)
parser.add_argument("--category", type=str, default="broad", help="Category type to use (broad or detailed)")
args = parser.parse_args()
exp_results = args.exp_results

sorted_args = sorted(vars(args).items())
for arg, value in sorted_args:
    # Print argument names in blue and values in green for clear differentiation
    print(f"\033[34m{arg}\033[0m: {value}")
input("Press Enter to continue...")
dataset_json_path = "datasets/ShareGPT4V/data/sharegpt4v/Dict_DSG_nondupid_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json"
with open(dataset_json_path) as f:
    data_all = json.load(f)
f.close()


global_output= dict()
for exp_result in tqdm(exp_results):
    quanti_result_path = os.path.join(exp_result_path,"quantitative results",f"evalscore_{exp_result.replace('/','_')}.json")
    model_name = exp_result.replace('/','_')
    
    with open(quanti_result_path, "r") as f:
        score_data = json.load(f)
    f.close()

    results_dict = dict()

    for folder, folder_resultdict in score_data.items():
        results_dict[folder] = dict()
        for k, v in folder_resultdict.items():
            item = data_all[folder][k]
            for q in item['question_list']:
                id = q["proposition_id"]
                if args.category == "detailed":
                    score_cat = q["category_detailed"]
                else:
                    score_cat = q["category_broad"]
                score = v[1][id]
                if score_cat not in results_dict[folder]:
                    results_dict[folder][score_cat] = [score]
                else:
                    results_dict[folder][score_cat].append(score)
                    
    folder_level_output = dict()
    output_overall = dict()
    output_folderagnostic = dict()
    
    for folder, folder_resultdict in results_dict.items():
        output_overall[folder] = dict()
        all = 0
        count = 0
        # folder별 category_detailed(k)별 평균이 나옴.
        for k,v in folder_resultdict.items():
            output_overall[folder][k] = np.mean(v)
            all += np.sum(v)
            count += len(v)
            if k not in output_folderagnostic:
                output_folderagnostic[k] = v
            else:
                output_folderagnostic[k].extend(v)
        folder_level_output[folder] = all / count

    all = 0
    count = 0
    for k,v in output_folderagnostic.items():
        output_folderagnostic[k] = np.mean(v)
        all += np.sum(v)
        count += len(v)
    folder_level_output['total'] = all / count

    # for k,v in folder_level_output.items():
    #     print(k)
    # for k,v in folder_level_output.items():
    #     print(v)

    for folder, score_type in output_overall.items():
        output_overall[folder]['all_type'] = folder_level_output[folder]
    output_overall['all_folder']={}
    for typ,scores in output_folderagnostic.items():
        output_overall['all_folder'][typ] = np.mean(scores)
    output_overall['all_folder']['all_type'] = folder_level_output['total']
    
    global_output[model_name] = output_overall
    
# Save output dictionary to CSV
with open(os.path.join(exp_result_path,'output_scores.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model_name', 'folder_name', 'category_detailed', 'score'])
    for model_name, output_overall in global_output.items():
        for folder, categories in output_overall.items():
            for category, score in categories.items():
                writer.writerow([model_name, folder, category, score])