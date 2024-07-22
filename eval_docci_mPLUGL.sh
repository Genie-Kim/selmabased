#!/bin/bash

# cannot run with multi gpu, so you can run with single gpu
# sudo nvidia-smi -pm 1
# img gen mergy for single gpu, but evaluation can use multigpu
# CUDA_VISIBLE_DEVICES=0 bash eval_docci_mPLUGL.sh "stabilityai/stable-diffusion-2-1" "clip_normal" 16
# CUDA_VISIBLE_DEVICES=0,1,2 bash eval_docci_mPLUGL.sh "stabilityai/stable-diffusion-2-1" "clip_normal" 16


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}

MODEL_ID=${1}
PIPENAME=${2}
BATCH_SIZE=${3}
DATA_DIR=$PWD
SPLITS=("test" "qual_test" "qual_dev")
# EXP_ROOT="exp_results"
# Q_JSONPATH="docci_meta_errorfix/docci_metadata_refined.jsonlines"


# Run the first Python script
# It can't run with multi gpu yet.

# =====================================
# Task 1: image generation
# =====================================
# CUDA_VISIBLE_DEVICES=${GPULIST[0]} python img_gen_merge.py \
# --pipename ${PIPENAME} \
# --batch_size ${BATCH_SIZE} \
# --model_id ${MODEL_ID} \
# --splits ${SPLITS[@]}


# =====================================
# Task 2: score generation with vqa model
# =====================================
MODEL_NAME=$(basename ${MODEL_ID})
# Check if the first script was successful
if [ $? -eq 0 ]; then
    # Run the second Python script
    python DSG/evaluation_mPLUG_multi.py \
    --worker_num_pergpu 7 \
    --exp_path "${MODEL_NAME}/${PIPENAME}_output" \
    --gpu_num ${GPULIST[@]} \
    --splits ${SPLITS[@]}
else
    echo "\n\nimg_gen_merge.py failed to execute successfully."
fi

