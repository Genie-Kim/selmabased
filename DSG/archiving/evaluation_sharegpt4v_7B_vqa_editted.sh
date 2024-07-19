#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/vqav2.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash DSG/evaluation_sharegpt4v_7B_vqa.sh

# cannot run with multi gpu, so you can run with single gpu
# sudo nvidia-smi -pm 1
# CUDA_VISIBLE_DEVICES=0 bash DSG/scripts/evaluation_sharegpt4v_7B_vqa_editted.sh "stable-diffusion-2-1/clip_repeat_textenc_True"

EXP_PATH=${1}
DATA_DIR=$PWD
EXP_ROOT="exp_results"
MODEL_NAME=${EXP_PATH//"/"/"_"}
Q_JSONPATH=${DATA_DIR}/datasets/ShareGPT4V/data/sharegpt4v/Dict_DSG_nondupid_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json

# make question file
python DSG/make_questionfile2share4v.py --exp_path ${EXP_PATH} \
--questionjsonpath ${Q_JSONPATH} \
--exp_root ${EXP_ROOT}

CKPT="Lin-Chen/ShareGPT4V-7B"
SPLIT=${MODEL_NAME}_questionFORshare4veval

# if you want to kill : alkil model_vqa_loader
python -m share4v.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file $EXP_ROOT/$SPLIT.jsonl \
    --image-folder $DATA_DIR \
    --answers-file $EXP_ROOT/answers_${SPLIT}.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

output_file=$EXP_ROOT/answers_${SPLIT}.jsonl

python DSG/make_answerfile2score.py --answer_file ${output_file} \
--org_jsonpath ${Q_JSONPATH} \
--exp_root ${EXP_ROOT} \
--exp_path ${EXP_PATH}