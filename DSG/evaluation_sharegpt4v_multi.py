import csv
import os
from PIL import Image
from vqa_utils import MPLUG
import json
from tqdm import tqdm
import numpy as np
import argparse
import multiprocessing

exp_result_path ="exp_results"

parser = argparse.ArgumentParser()
parser.add_argument('--exp_path', type=str, default="stable-diffusion-v1-5/clip_repeat_textenc_False")
parser.add_argument('--worker_num', type=int, default=2)
parser.add_argument('--cache_dir', type=str, default="pretrained_models")
args = parser.parse_args()

args.image_path = os.path.join(exp_result_path,args.exp_path)
image_path = args.image_path
args.save_path = os.path.join(exp_result_path,f"evalscore_{args.exp_path.replace('/','_')}.json")
save_path = args.save_path

os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
os.environ['MODELSCOPE_CACHE'] = args.cache_dir

sorted_args = sorted(vars(args).items())
for arg, value in sorted_args:
    # Print argument names in blue and values in green for clear differentiation
    print(f"\033[34m{arg}\033[0m: {value}")
input("Press Enter to continue...")


def divide_input_tuple_list(input_tuple_list, worker_num):
    k, m = divmod(len(input_tuple_list), worker_num)
    return [input_tuple_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(worker_num)]

def get_answer(input_sub_list):
    outputdict = dict()
    vqa_model = MPLUG()
    input_num =len(input_sub_list)
    VQA = vqa_model.vqa
    for idx, question_list, img_path in input_sub_list:
        for question in question_list:
            question = question['question_natural_language']
            generated_image = Image.open(img_path)
            answer = VQA(generated_image, question)
            if idx not in outputdict:
                outputdict[idx] = {}
            outputdict[idx][question] =answer
        print(f"{len(outputdict)}/{input_num}")
    return outputdict


with open("datasets/ShareGPT4V/data/sharegpt4v/Dict_DSG_nondupid_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json") as f:
    data_all = json.load(f)
f.close()

input_tuple_list = []
for folder, folderdict in data_all.items():
    for idx, v in folderdict.items():
        input_tuple_list.append((idx, v['question_list'], os.path.join(image_path, f"{idx}.jpg")))

divided_input_list=divide_input_tuple_list(input_tuple_list,args.worker_num)

with multiprocessing.Pool(args.worker_num) as p:
    total_dict_list = list(p.imap(get_answer, divided_input_list))


answers_dict = {}
for worker_dict in total_dict_list:
    for idx, QkeyAdict in worker_dict.items():
        answers_dict[idx] = QkeyAdict
 

results = dict()
for folder, folderdict in data_all.items():
    results[folder] = dict()
    for idx, v in tqdm(folderdict.items()):
        id2scores = dict()
        id2dependency = dict()
        for item in v['question_list']:
            id = item["proposition_id"]
            question = item["question_natural_language"]
            answer = answers_dict[idx][question]
            id2scores[str(id)] = float(answer == 'yes')
            id2dependency[str(id)] = [xx.strip() for xx in str(item["dependency"]).split(",")]

        for id, parent_ids in id2dependency.items():
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == '0':
                    continue
                if parent_id in id2scores and id2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                id2scores[id] = 0 # 부모 중 하나라도 no를 선택했으면 자식도 no로 처리

        average_score = sum(id2scores.values()) / len(id2scores)

        results[folder][idx] = [average_score, id2scores]

with open(save_path, "w") as f:
    json.dump(results, f)
f.close()

with open(save_path, "r") as f:
    score_data = json.load(f)
f.close()

results_dict = dict()

for folder, folder_resultdict in score_data.items():
    results_dict[folder] = dict()
    for k, v in folder_resultdict.items():
        item = data_all[folder][k]
        for q in item['question_list']:
            id = q["proposition_id"]
            score_cat = q["category_detailed"]
            score = v[1][id]
            if score_cat not in results_dict[folder]:
                results_dict[folder][score_cat] = [score]
            else:
                results_dict[folder][score_cat].append(score)

output = dict()
output_folderagnostic = dict()
for folder, folder_resultdict in results_dict.items():
    output[folder] = dict()
    all = 0
    count = 0
    # folder별 category_detailed(k)별 평균이 나옴.
    for k,v in folder_resultdict.items():
        output[folder][k] = np.mean(v)
        all += np.sum(v)
        count += len(v)
        if k not in output_folderagnostic:
            output_folderagnostic[k] = v
        else:
            output_folderagnostic[k].extend(v)
    print(f"{folder} Average", all / count)
    print(output[folder])


all = 0
count = 0
for k,v in output_folderagnostic.items():
    output_folderagnostic[k] = np.mean(v)
    all += np.sum(v)
    count += len(v)
print("\nTotal Average (folder agnostic)", all / count)
print(output_folderagnostic)

