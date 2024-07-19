#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/vqav2.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash DSG/evaluation_sharegpt4v_7B_vqa.sh
# cannot run with multi gpu, so you can run with single gpu
# CUDA_VISIBLE_DEVICES=0 bash DSG/evaluation_sharegpt4v_7B_vqa.sh


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXP_PATH=${1}
DATA_DIR=$PWD
EXP_ROOT="exp_results"
MODEL_NAME=${EXP_PATH//"/"/"&"}
Q_JSONPATH="datasets/docci/docci_metadata.jsonlines"

# make question file
python DSG/make_questionfile2share4v.py --exp_path ${EXP_PATH} \
--questionjsonpath ${Q_JSONPATH} \
--splits "test" "qual_test" "qual_dev"

CKPT="Lin-Chen/ShareGPT4V-7B"
SPLIT="${MODEL_NAME}&questionFORshare7Beval"
OUTDIR="${MODEL_NAME}&AnswerOFshare7Beval"

# if you want to kill : alkil model_vqa_loader
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m share4v.eval.model_vqa_loader \
        --model-path ${CKPT} \
        --question-file $EXP_ROOT/$SPLIT.jsonl \
        --image-folder $DATA_DIR \
        --answers-file $EXP_ROOT/$OUTDIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$EXP_ROOT/$OUTDIR/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EXP_ROOT/$OUTDIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done



# python DSG/make_answerfile2score.py --answer_file ${output_file} \
# --org_jsonpath ${Q_JSONPATH} \
# --exp_root ${EXP_ROOT} \
# --exp_path ${EXP_PATH}