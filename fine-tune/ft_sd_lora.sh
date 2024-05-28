export MODEL_NAME="stabilityai/stable-diffusion-2"
export TRAIN_DIR="llm/Image_Data/SD\ v2/localized_narrative_sd_1k/localized_narrative_sd_1k"


accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="/home/compu/JinProjects/jinprojects/SELMA/llm/Image_Data/SD v2/localized_narrative_sd_1k/localized_narrative_sd_1k" \
  --resolution=512 --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --checkpointing_steps 500 --checkpoints_total_limit 20 \
  --learning_rate=3e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank 128 \
  --output_dir="checkpoints/test" \
  --report_to="wandb"