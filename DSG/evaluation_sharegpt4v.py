import csv
import os
from PIL import Image
from vqa_utils import MPLUG
import json
from tqdm import tqdm
import numpy as np
import argparse
os.environ["TRANSFORMERS_CACHE"] = 'pretrained_models'
os.environ['MODELSCOPE_CACHE'] = 'pretrained_models'

exp_result_path ="exp_results"
parser = argparse.ArgumentParser()
parser.add_argument('--exp_path', type=str, default="stable-diffusion-v1-5/clip_repeat_textenc_False")
args = parser.parse_args()
image_path = os.path.join(exp_result_path,args.exp_path)
save_path = os.path.join(exp_result_path,f"evalscore_{args.exp_path.replace('/','_')}.json")


with open("datasets/ShareGPT4V/data/sharegpt4v/Dict_DSG_nondupid_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json") as f:
    data_all = json.load(f)
f.close()

print("Loading mPLUG-large")
vqa_model = MPLUG()
results = dict()
for folder, folderdict in data_all.items():
    results[folder] = dict()
    for idx, v in tqdm(folderdict.items()):
        text = v['gpt4vcaption']
        # assume dataset id(key) same as image name
        generated_image = Image.open(os.path.join(image_path, f"{idx}.jpg"))
        VQA = vqa_model.vqa
        id2scores = dict()
        id2dependency = dict()
        for item in v['question_list']:
            id = item["proposition_id"]
            answer = VQA(generated_image, item["question_natural_language"])
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

