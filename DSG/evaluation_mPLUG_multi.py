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
original_image_path = "exp_results/docci/images_eval_output"

# if you want to use the original image, you should set the exp_path to "docci/image_eval_output".

def parse_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_path", type=str, required=True, help="Path to the experiment results file"
    )
    parser.add_argument("--worker_num_pergpu", type=int, default=7)
    parser.add_argument("--gpu_num", nargs="+",type=str, default=["0"])
    parser.add_argument("--cache_dir", type=str, default="pretrained_models")
    parser.add_argument('--questionjsonpath', type=str, default="docci_meta_errorfix/docci_metadata_refined.jsonlines", help='Path to the dataset JSON file')
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
    
    args.save_path = os.path.join(quantipath,f"evalscore&{args.exp_path.replace('/','&')}&mpluglarge_origin.json")

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
        if args.exp_path == 'docci/images_eval_output':
            image_path = os.path.join(
                original_image_path,split,f"{example_id}.jpg"
            )
        else:
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

    # =====================================
    # Task 2: convert answer to scores (with dsg calculation)
    # =====================================


    # answerpath = os.path.join(
    #     exp_root_path,
    #     "quantitative results",
    #     f"answersdict&{args.exp_path.replace('/','&')}&mpluglarge.json")
    
    # with open(answerpath, "r") as f:
    #     answers_dict = json.load(f)
    
    results = dict()    

    for example_id, values in tqdm(docci_id2meta.items()):
        split = '_'.join(example_id.split('_')[:-1])
        
        if split in args.splits:
            question_all_sentences = values['dsg']['question']
            for sentence_id, sent_dict in question_all_sentences.items():
                questions = sent_dict['question']
                if len(questions)==0:
                    continue
                else:
                    id2dependency = sent_dict['dependency']
                    id2scores = dict()
                    for question_id,question in questions.items():
                        answer_key = f"{example_id}&{sentence_id}&{question_id}"
                        answer = answers_dict[answer_key]
                        id2scores[question_id] = float("yes" in answer)

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

                    average_score = sum(id2scores.values()) / len(id2scores) # sentence average score
                    sentence_key = f"{example_id}&{sentence_id}"
                    results[sentence_key] = [average_score, id2scores]


    # SAVE score file.
    with open(args.save_path, "w") as f:
        json.dump(results, f)

    # =====================================
    # Task 3: refine the scores
    # =====================================

    # with open(args.save_path, "r") as f:
    #     score_data = json.load(f)
    # f.close()

    score_summ = dict()
    for example_id, values in tqdm(docci_id2meta.items()):
        split = '_'.join(example_id.split('_')[:-1])
        if split not in score_summ:
            score_summ[split]={}
        if 'dsg' in values:
            for sent_id, sent_info in docci_id2meta[example_id]['dsg']['question'].items():
                score_key = f"{example_id}&{sent_id}"
                
                for qid, question in sent_info['question'].items():
                    cat_broad = sent_info['cat_broad'][qid]
                    cat_detail = sent_info['cat_detail'][qid]
                    if cat_broad not in score_summ[split]:
                        score_summ[split][cat_broad]={}
                    if cat_detail not in score_summ[split][cat_broad]:
                        score_summ[split][cat_broad][cat_detail]=[]
                    
                    question_score = results[score_key][1][qid]
                    score_summ[split][cat_broad][cat_detail].append(question_score)
                    
    output_csv = dict()
    for split, cat_broad_dict in score_summ.items():
        all = 0
        count = 0
        for cat_braod, cat_detail_dict in cat_broad_dict.items():
            cat_broad_all=0
            cat_broad_count=0
            for cat_detail, values in cat_detail_dict.items():
                csv_key = f"{split}&{cat_braod}&{cat_detail}"
                output_csv[csv_key] = np.mean(values)
                cat_broad_all += np.sum(values)
                cat_broad_count += len(values)
                all += np.sum(values)
                count += len(values)
            csv_key = f"{split}&{cat_braod}&Total"
            output_csv[csv_key] = cat_broad_all / cat_broad_count
        csv_key=f"{split}&Total&Total"
        output_csv[csv_key] = all / count
                
                
                
    model_name = args.exp_path.replace("/", "&")
    
    sdversion = model_name.split("&")[0]
    sdversion = sdversion.replace("stable-diffusion-2-1","SD2.1")
    sdversion = sdversion.replace("stable-diffusion-v1-5","SD1.5")
    pipename = model_name.split("&")[1]
    pipename = "_".join(pipename.split("_")[:-1])
    
    csv_path = os.path.join(
        exp_root_path,
        "quantitative results",
        f"evalscore&{model_name}&mpluglarge_summary.csv",
    )

    # Save output dictionary to CSV
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model name","method", "split", "category_broad", "category_detailed", "score","information type"])

        for csv_key, score in output_csv.items():
            split = csv_key.split('&')[0]
            cat_broad = csv_key.split('&')[1]
            cat_detail = csv_key.split('&')[2]
            if cat_detail in ['count']:
                information_type = 'Count'
            elif cat_detail in ['size', 'shape']:
                information_type = 'Size&shape'
            elif cat_detail in ['color', 'texture', 'material']:
                information_type = 'Style'
            elif cat_detail in ['spatial']:
                information_type = "Spatial"
            else:
                information_type="None"
                
            writer.writerow([sdversion,pipename, split, cat_broad, cat_detail,score,information_type])
            
    # calculate token length per score. results[score_key][1][qid]
    token2score = {}
    for i in range(2,571):
        token2score[i] = []
        
    for example_id, values in tqdm(docci_id2meta.items()):
        if 'dsg' in values:
            for sent_id, sent_info in docci_id2meta[example_id]['dsg']['question'].items():
                if len(sent_info['question'])>0:
                    score_key = f"{example_id}&{sent_id}"
                    sent_avgscore = results[score_key][0]
                    for i in token2score.keys():
                        if sent_info['start_tokenlen']<=i and i<=sent_info['end_tokenlen']:
                            token2score[i].append(sent_avgscore)
    tokenlen_csv_path = os.path.join(
        exp_root_path,
        "quantitative results",
        f"evalscore&{model_name}&mpluglarge_tokenlen.csv",
    )
    with open(tokenlen_csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model name","method", "token position", "average score","# of sentences on this position"])
        for token_position, scores in token2score.items():
            count = len(scores)
            if len(scores)==0:
                scores =[0]
            writer.writerow([sdversion,pipename, token_position, np.mean(scores), count])
