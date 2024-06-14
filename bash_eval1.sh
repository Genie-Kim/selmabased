#!/bin/bash

# Run the first Python script
CUDA_VISIBLE_DEVICES=3 python img_gen_merge.py --text_enc "clip" \
--repeat_textenc \
--batch_size 1 \
--model_id "stabilityai/stable-diffusion-2-1"

# Check if the first script was successful
if [ $? -eq 0 ]; then
    # Run the second Python script
    CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 7 \
    --exp_path "stable-diffusion-2-1/clip_repeat_textenc_True"
else
    echo "img_gen_merge.py failed to execute successfully."
fi




CUDA_VISIBLE_DEVICES=0 python img_gen_merge.py --text_enc "clip" \
--batch_size 16 \
--model_id "stabilityai/stable-diffusion-2-1"

# not done
CUDA_VISIBLE_DEVICES=0 python DSG/evaluation_sharegpt4v_multi.py --worker_num 7 \
--exp_path "stable-diffusion-2-1/clip_repeat_textenc_False"


