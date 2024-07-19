import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import csv

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset JSON path.')
    parser.add_argument('--answer_file', type=str, required=True, help='Path to the dataset JSON file')
    parser.add_argument('--org_jsonpath', type=str, default="datasets/docci/docci_metadata.jsonlines", help='Path to the experiment results file')
    parser.add_argument('--exp_root', type=str, default="exp_results", help='Path to the experiment results file')
    parser.add_argument('--exp_path', type=str, required=True, help='Path to the model file')
    return parser.parse_args()

# Main function
def main():
    args = parse_arguments()
    
    sorted_args = sorted(vars(args).items())
    for arg, value in sorted_args:
        # Print argument names in blue and values in green for clear differentiation
        print(f"\033[34m{arg}\033[0m: {value}")
    
    model_name = args.exp_path.replace('/','&')
    
    save_name = f"evalscore&{model_name}&sharegpt4v7B.json"
    save_path = os.path.join(args.exp_root,save_name)

    # Load the JSON data
    docci_id2meta={}
    with open(args.org_jsonpath, "r") as f:
        for line in f:
            item = json.loads(line)
            key = item.pop("example_id")
            docci_id2meta[key] = item
            
    
    answers = [json.loads(q) for q in open(
        os.path.expanduser(args.answer_file), "r")]
    
    answers_qid2ans = {q["question_id"]: q["text"].split(' ')[0].lower() for q in answers}
    
    # TODO: Not done following part.  exp_results/stable-diffusion-2-1&clip_normal_output&&AnswerOFshare7Beval/merge.jsonl
    results = dict()
    for folder, folderdict in data_all.items():
        results[folder] = dict()
        for idx, v in tqdm(folderdict.items()):
            id2scores = dict()
            id2dependency = dict()
            for qid, item in enumerate(v["question_list"]):
                id = item["proposition_id"]
                qid = idx + "##@" + str(qid)
                answer = answers_qid2ans[qid]
                id2scores[str(id)] = float("yes" in answer)
                id2dependency[str(id)] = [
                    xx.strip() for xx in str(item["dependency"]).split(",")
                ]

            for id, parent_ids in id2dependency.items():
                any_parent_answered_no = False
                for parent_id in parent_ids:
                    if parent_id == "0":
                        continue
                    if parent_id in id2scores and id2scores[parent_id] == 0:
                        any_parent_answered_no = True
                        break
                if any_parent_answered_no:
                    id2scores[id] = 0  # 부모 중 하나라도 no를 선택했으면 자식도 no로 처리

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
            for q in item["question_list"]:
                id = q["proposition_id"]
                # score_cat = q["category_detailed"]
                score_cat = q["category_broad"]
                score = v[1][id]
                if score_cat not in results_dict[folder]:
                    results_dict[folder][score_cat] = [score]
                else:
                    results_dict[folder][score_cat].append(score)

    global_output = dict()
    output = dict()
    output_folderagnostic = dict()

    for folder, folder_resultdict in results_dict.items():
        output[folder] = dict()
        all = 0
        count = 0
        # folder별 category_detailed(k)별 평균이 나옴.
        for k, v in folder_resultdict.items():
            output[folder][k] = np.mean(v)
            all += np.sum(v)
            count += len(v)
            if k not in output_folderagnostic:
                output_folderagnostic[k] = v
            else:
                output_folderagnostic[k].extend(v)
        global_output[folder] = all / count

    all = 0
    count = 0
    for k, v in output_folderagnostic.items():
        output_folderagnostic[k] = np.mean(v)
        all += np.sum(v)
        count += len(v)
    global_output["total"] = all / count

    for folder, score_type in output.items():
        output[folder]["all_type"] = global_output[folder]
    output["all_folder"] = {}
    for typ, scores in output_folderagnostic.items():
        output["all_folder"][typ] = np.mean(scores)
    output["all_folder"]["all_type"] = global_output["total"]

    # Save output dictionary to CSV
    with open(os.path.join(exp_root,f"{model_name} output_scores_sharegpt4v7B.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model name", "folder name", "category_detailed", "score"])
        for folder, categories in output.items():
            for category, score in categories.items():
                writer.writerow([model_name, folder, category, score])


if __name__ == "__main__":
    main()
