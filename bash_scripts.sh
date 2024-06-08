CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 4 \
--exp_path "stable-diffusion-v1-5/clip_repeat_textenc_True"

CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 4 \
--exp_path "stable-diffusion-v1-5/clip_repeat_textenc_False"

CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 4 \
--exp_path "stable-diffusion-v1-5/longclip_repeat_textenc_False"

CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 4 \
--exp_path "stable-diffusion-2/clip_repeat_textenc_False"

CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 4 \
--exp_path "stable-diffusion-2/clip_repeat_textenc_True"



# tm04
CUDA_VISIBLE_DEVICES=3 python img_gen_merge.py --text_enc "clip" \
--repeat_textenc \
--batch_size 1 \
--model_id "stabilityai/stable-diffusion-2"

# tm03
CUDA_VISIBLE_DEVICES=2 python img_gen_merge.py --text_enc "clip" \
--batch_size 16 \
--model_id "stabilityai/stable-diffusion-2"

# tm02
CUDA_VISIBLE_DEVICES=1 python img_gen_merge.py --text_enc "clip" \
--batch_size 16 \
--model_id "runwayml/stable-diffusion-v1-5"

# tm01
CUDA_VISIBLE_DEVICES=0 python img_gen_merge.py --text_enc "longclip" \
--batch_size 16 \
--model_id "runwayml/stable-diffusion-v1-5"

# tm05
CUDA_VISIBLE_DEVICES=3 python img_gen_merge.py --text_enc "clip" \
--repeat_textenc \
--batch_size 1 \
--model_id "runwayml/stable-diffusion-v1-5"










