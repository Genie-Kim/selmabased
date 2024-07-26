
tmux send-keys -t tm01 'CUDA_VISIBLE_DEVICES=0 bash eval_docci_mPLUGL.sh "runwayml/stable-diffusion-v1-5" "clip_repeattext_30" 1' ENTER
tmux send-keys -t tm02 'CUDA_VISIBLE_DEVICES=1 bash eval_docci_mPLUGL.sh "runwayml/stable-diffusion-v1-5" "clip_repeattext_40" 1' ENTER
tmux send-keys -t tm03 'CUDA_VISIBLE_DEVICES=2 bash eval_docci_mPLUGL.sh "runwayml/stable-diffusion-v1-5" "clip_repeattext_50" 1' ENTER

#######################

# tmux send-keys -t tm07 'CUDA_VISIBLE_DEVICES=3 bash eval_docci_mPLUGL.sh "docci" "oracle" 1' ENTER

#######################

# tmux send-keys -t tm07 'CUDA_VISIBLE_DEVICES=3 bash eval_docci_mPLUGL.sh "runwayml/stable-diffusion-v1-5" "longclip" 1' ENTER

# tmux send-keys -t tm01 'CUDA_VISIBLE_DEVICES=0 bash eval_docci_mPLUGL.sh "stabilityai/stable-diffusion-2-1" "clip_normal" 1' ENTER
# tmux send-keys -t tm03 'CUDA_VISIBLE_DEVICES=1 bash eval_docci_mPLUGL.sh "stabilityai/stable-diffusion-2-1" "clip_summary" 1' ENTER
# tmux send-keys -t tm05 'CUDA_VISIBLE_DEVICES=2 bash eval_docci_mPLUGL.sh "runwayml/stable-diffusion-v1-5" "clip_repeattext" 1' ENTER

# tmux send-keys -t tm02 'CUDA_VISIBLE_DEVICES=0 bash eval_docci_mPLUGL.sh "stabilityai/stable-diffusion-2-1" "clip_repeattext" 1' ENTER
# tmux send-keys -t tm04 'CUDA_VISIBLE_DEVICES=1 bash eval_docci_mPLUGL.sh "runwayml/stable-diffusion-v1-5" "clip_summary" 1' ENTER
# tmux send-keys -t tm06 'CUDA_VISIBLE_DEVICES=2 bash eval_docci_mPLUGL.sh "runwayml/stable-diffusion-v1-5" "clip_normal" 1' ENTER


# ####################### 

# CUDA_VISIBLE_DEVICES=2 python img_gen_merge.py \
# --pipename "clip_summary" \
# --batch_size 16 \
# --model_id "stabilityai/stable-diffusion-2-1"


# CUDA_VISIBLE_DEVICES=0 python img_gen_merge.py \
# --pipename "clip_summary" \
# --batch_size 16 \
# --model_id "runwayml/stable-diffusion-v1-5"





# #######################

# # tm02
# CUDA_VISIBLE_DEVICES=0 python img_gen_merge.py \
# --pipename "clip_normal" \
# --batch_size 16 \
# --model_id "runwayml/stable-diffusion-v1-5"

# # tm03
# CUDA_VISIBLE_DEVICES=1 python img_gen_merge.py \
# --pipename "longclip" \
# --batch_size 16 \
# --model_id "runwayml/stable-diffusion-v1-5"

# # tm04
# CUDA_VISIBLE_DEVICES=2 python img_gen_merge.py \
# --pipename "clip_repeattext" \
# --batch_size 1 \
# --model_id "runwayml/stable-diffusion-v1-5"

# # tm05
# CUDA_VISIBLE_DEVICES=2 python img_gen_merge.py \
# --pipename "clip_repeattext" \
# --batch_size 1 \
# --model_id "stabilityai/stable-diffusion-2-1"

# # tm06
# CUDA_VISIBLE_DEVICES=3 python img_gen_merge.py \
# --pipename "clip_normal" \
# --batch_size 16 \
# --model_id "stabilityai/stable-diffusion-2-1"


# #######################




# # not done
# CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 7 \
# --exp_path "stable-diffusion-2/clip_repeat_textenc_True"

# # tm01
# CUDA_VISIBLE_DEVICES=3 python DSG/evaluation_sharegpt4v_multi.py --worker_num 7 \
# --exp_path "stable-diffusion-v1-5/clip_repeat_textenc_True"

# # tm02
# CUDA_VISIBLE_DEVICES=2 python DSG/evaluation_sharegpt4v_multi.py --worker_num 7 \
# --exp_path "stable-diffusion-v1-5/clip_repeat_textenc_False"

# # tm03
# CUDA_VISIBLE_DEVICES=1 python DSG/evaluation_sharegpt4v_multi.py --worker_num 7 \
# --exp_path "stable-diffusion-v1-5/longclip_repeat_textenc_False"

# #tm04
# CUDA_VISIBLE_DEVICES=0 python DSG/evaluation_sharegpt4v_multi.py --worker_num 7 \
# --exp_path "stable-diffusion-2/clip_repeat_textenc_False"



    # choices=[
    #     "regionaldiffusion",
    #     "longclip",
    #     "clip_repeattext",
    #     "clip_normal",
    #     "clip_summary",
    #     "abstractdiffusion",
    # ],

# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "runwayml/stable-diffusion-v1-5"











