import argparse
import csv
import json
import multiprocessing
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from vqa_utils import MPLUG
import torch

exp_root_path = "exp_results"


def parse_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_path", type=str, required=True, help="Path to the experiment results file"
    )
    parser.add_argument("--worker_num_pergpu", type=int, default=7)
    parser.add_argument("--gpu_num", nargs="+",type=str, default=["0"])
    parser.add_argument("--cache_dir", type=str, default="pretrained_models")
    parser.add_argument('--questionjsonpath', type=str, default="datasets/docci/docci_metadata.jsonlines", help='Path to the dataset JSON file')
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
        default=["test","qual_test","qual_dev"],
    )
    args = parser.parse_args()

    quantipath = os.path.join(
        exp_root_path,
        "quantitative results")
    os.makedirs(quantipath, exist_ok=True)
    
    args.save_path = os.path.join(quantipath,f"evalscore&{args.exp_path.replace('/','&')}&mpluglarge.json",
    )

    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["MODELSCOPE_CACHE"] = args.cache_dir

    sorted_args = sorted(vars(args).items())
    for arg, value in sorted_args:
        # Print argument names in blue and values in green for clear differentiation
        print(f"\033[34m{arg}\033[0m: {value}")
    # input("Press Enter to continue...")
    
    return args


def divide_input_tuple_list(input_tuple_list, worker_num):
    k, m = divmod(len(input_tuple_list), worker_num)
    return [
        input_tuple_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(worker_num)
    ]


def get_answer(input_sub_list_gpunum):
    input_sub_list = input_sub_list_gpunum[0]
    gpu_num = input_sub_list_gpunum[1]
    
    outputdict = dict()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    
    vqa_model = MPLUG()
    VQA = vqa_model.vqa
    for idx, question_dict, img_path in tqdm(input_sub_list):
        for qid, question in question_dict.items():
            question_id = f"{idx}&{qid}"
            generated_image = Image.open(img_path)
            answer = VQA(generated_image, question)
            if question_id not in outputdict:
                outputdict[question_id] = {}
            outputdict[question_id] = answer
    return outputdict

if __name__ == "__main__":
    args = parse_argments()

    # Load the JSON data
    docci_id2meta={}
    with open(args.questionjsonpath, "r") as f:
        for line in f:
            item = json.loads(line)
            if 'dsg' in item:
                key = item.pop("example_id")
                docci_id2meta[key] = item

    # =====================================
    # Task 1: generate answer
    # =====================================

    model_exppath = os.path.join(exp_root_path, args.exp_path)
    input_tuple_list = []
    # Process the data (example)
    print("prepareing multi-processing")
    for example_id, values in tqdm(docci_id2meta.items()):
        split = '_'.join(example_id.split('_')[:-1])
        image_path = os.path.join(model_exppath,split,f"{example_id}.jpg")
        assert os.path.isfile(image_path), f"there is no image file in {image_path}"
        
        if split in args.splits:
            question_all_sentences = values['dsg']['question']
            for sentence_id, ques_info in question_all_sentences.items():
                idx = f"{example_id}&{sentence_id}"
                questions = ques_info['question']
                input_tuple_list.append((idx, questions, image_path))

    divided_input_list = divide_input_tuple_list(input_tuple_list, args.worker_num_pergpu*len(args.gpu_num))    
    for i,gpunum in enumerate(args.gpu_num):
        for j in range(len(divided_input_list)):
            if i*args.worker_num_pergpu-1<j and j<(i+1)*args.worker_num_pergpu:
                divided_input_list[j] = (divided_input_list[j],gpunum)
            
    with multiprocessing.Pool(args.worker_num_pergpu*len(args.gpu_num)) as p:
        total_dict_list = list(p.imap(get_answer, divided_input_list))
    
    # get_answer(divided_input_list[0]) # for debug

    answers_dict = {}
    for worker_dict in total_dict_list:
        for idx, QkeyAdict in worker_dict.items():
            answers_dict[idx] = QkeyAdict


    answerpath = os.path.join(
        exp_root_path,
        "quantitative results",
        f"answersdict&{args.exp_path.replace('/','&')}&mpluglarge.json")
    
    with open(answerpath, "w") as f:
        json.dump(answers_dict, f, indent=4)
    f.close()

    # # =====================================
    # # Task 2: convert answer to scores (with dsg calculation)
    # # =====================================

    # results = dict()    

    # for example_id, values in tqdm(docci_id2meta.items()):
    #     split = '_'.join(example_id.split('_')[:-1])
        
    #     if split in args.splits:
    #         question_all_sentences = values['dsg']['question']
    #         for sentence_id, ques_info in question_all_sentences.items():
    #             questions = ques_info['question']
    #             if len(questions)==0:
    #                 continue
    #             else:
    #                 id2dependency = ques_info['dependency']
    #                 id2scores = dict()
    #                 for question_id,question in questions.items():
    #                     answer_key = f"{example_id}&{sentence_id}&{question_id}"
    #                     answer = answers_dict[answer_key]
    #                     id2scores[question_id] = float("yes" in answer)
    #                     id2dependency[question_id] = [
    #                         xx.strip() for xx in str(id2dependency[question_id]).split(",")
    #                     ]

    #                 for id, parent_ids in id2dependency.items():
    #                     any_parent_answered_no = False
    #                     for parent_id in parent_ids:
    #                         if parent_id == "0":
    #                             continue
    #                         if parent_id in id2scores and id2scores[parent_id] == 0:
    #                             any_parent_answered_no = True
    #                             break
    #                     if any_parent_answered_no:
    #                         id2scores[id] = 0  # 부모 중 하나라도 no를 선택했으면 자식도 no로 처리

    #                 average_score = sum(id2scores.values()) / len(id2scores) # sentence average score
    #                 sentence_key = f"{example_id}&{sentence_id}"
    #                 results[sentence_key] = [average_score, id2scores]

    # with open(args.save_path, "w") as f:
    #     json.dump(results, f)
    # f.close()

    # # =====================================
    # # Task 3: refine the scores
    # # =====================================

    # # with open(args.save_path, "r") as f:
    # #     score_data = json.load(f)
    # # f.close()

    # results_dict = dict()
    # for sentence_key, score_list in results.items():
    #     example_id= sentence_key.split("&")[0]
    #     sentence_id= sentence_key.split("&")[1]
        
    #     sentence_dict = docci_id2meta[example_id]['dsg']['question'][sentence_id]
    #     sentence = sentence_dict['prompt']
    #     for qid,question in sentence_dict['question']:
    #         qid_tuple = sentence_dict['tuple'][qid]
    #         category_broad = qid_tuple.split("-")[0].strip()
            
    #         score = score_list[1][qid]
    #         if category_broad not in results_dict:
    #             results_dict[category_broad] = [score]
    #         else:
    #             results_dict[category_broad].append(score)

    # global_output = dict()
    # output_folderagnostic = dict()
    # output = dict()

    # all = 0
    # count = 0
    # # folder별 category_detailed(k)별 평균이 나옴.
    # for k, v in results_dict.items():
    #     output[k] = np.mean(v)
    #     all += np.sum(v)
    #     count += len(v)
    #     if k not in output_folderagnostic:
    #         output_folderagnostic[k] = v
    #     else:
    #         output_folderagnostic[k].extend(v)
    # output["all_type"] = all / count
      
    # model_name = args.exp_path.replace("/", "&")

    # csv_path = os.path.join(
    #     exp_root_path,
    #     "quantitative results",
    #     f"{model_name}_scores_mpluglarge.csv",
    # )

    # # Save output dictionary to CSV
    # with open(csv_path, "w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["model name", "folder name", "category_detailed", "score"])
    #     for folder, categories in output.items():
    #         for category, score in categories.items():
    #             writer.writerow([model_name, folder, category, score])
