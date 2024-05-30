from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from models.LongCLIP import longclip
import shutil
import torch
import json
import csv
import os
from tqdm import tqdm
import argparse
from models.RepeatTextencStableDiffusion_base import RepeatTextencStableDiffusionPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument('--prompt_path', type=str, default="datasets/ShareGPT4V/data/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json")
parser.add_argument('--output_path', type=str, default="longtext_imggen")
parser.add_argument('--cache_dir', type=str, default="/home/compu/JinProjects/jinprojects/SELMA/pretrained_models")
parser.add_argument('--eval_benchmark', type=str, default="DSG")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--text_enc', type=str, default="clip")
parser.add_argument('--repeat_textenc', type=bool, default=False)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# model_id = "stabilityai/stable-diffusion-2"
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# model_id = "CompVis/stable-diffusion-v1-4"
# model_id = "runwayml/stable-diffusion-v1-5"
model_id = args.model_id
output_path = args.output_path
batch_size = args.batch_size
dataset_path = "/home/compu/JinProjects/jinprojects/SELMA/datasets/ShareGPT4V/data"

assert model_id!="stabilityai/stable-diffusion-2" or args.text_enc!='longclip', "LongCLIP model is not supported SD v2.1"

if args.text_enc == 'longclip':
    longclip_modelpath= os.path.join(args.cache_dir,"LongCLIP/longclip-L.pt")
    
else:
    longclip_modelpath= None


if "xl" in model_id:
    print("not implemented")
    #     pipe = RepeatTextencStableDiffusionPipeline.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float16,
    #         cache_dir=args.cache_dir,
    #         safety_checker=None,
    #         seed=2468,
    #         repeat_textenc=args.repeat_textenc,
    #     )
    
else:
    pipe = RepeatTextencStableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        safety_checker=None,
        seed=2468,
        repeat_textenc=args.repeat_textenc,
        longclip_modelfile=longclip_modelpath,
    )
    # # for match with rpg model's configuration.
    # num_inference_steps=20,  # sampling step
    # height=1024,
    # width=1024,
    # guidance_scale=7.0,

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

data_all = dict()

with open(args.prompt_path) as f:
    data = json.load(f)
f.close()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

##############

prompts = []
names = []
for i, item in enumerate(tqdm(data)):
    prompt = item['conversations'][1]['value']
    imageid = item['id']
    imagepath = os.path.join(dataset_path,item['image'])
    name = f"sharegpt4v_{imageid}"
    prompts.append(prompt)
    names.append(name)

    if len(prompts) == batch_size:
        images = pipe(prompts).images
        print(len(images))
        for j, img in enumerate(images):
            img.save(os.path.join(output_path, f"{names[j]}.jpg" ))
            # TODO : copy the image from the imagepath to the output path
            shutil.copy(imagepath, os.path.join(output_path, f"{names[j]}_original.jpg"))

        prompts = []
        names = []
    
    if i>10:
        break


#############

if args.eval_benchmark == "DSG":
    data_all = dict()
    with open("DSG/dsg-1k-anns.csv") as f:
        data = csv.reader(f, delimiter=",")
        next(data)
        for row in data:
            if row[0] in data_all:
                continue
            data_all[row[0]] = row[1]
    f.close()
    print(len(data_all))

    for k, v in tqdm(data_all.items()):
        name = k
        prompt = v
        images = pipe(prompt, num_images_per_prompt=1, cross_attention_kwargs={"scale": scale / 100}).images[0]
        images.save(os.path.join(output_path, f"{name}.jpg"))

elif args.eval_benchmark == "TIFA":
    with open("../TIFA/tifa_v1.0/tifa_v1.0_text_inputs.json", "r") as f:
        data_all = json.load(f)
    f.close()

    for item in tqdm(data_all):
        prompt = item["caption"]
        name = item["id"]
        images = pipe(prompt, num_images_per_prompt=1, cross_attention_kwargs={"scale": scale / 100}).images[0]
        images.save(os.path.join(output_path, f"{name}.jpg"))




# if args.text_enc == 'clip':
#     tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
#     text_encoder = CLIPTextModel.from_pretrained(
#         args.model_id, subfolder="text_encoder", use_safetensors=False
#     )

# def get_textemb(prompt):
    
#     text_input = tokenizer(
#         prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
#     )

#     # Detokenize text_input
#     # detokenized_text = tokenizer.decode(text_input.input_ids[0])

#     with torch.no_grad():
#         text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        
#     max_length = text_input.input_ids.shape[-1]
#     uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
#     uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

#     return torch.cat([uncond_embeddings, text_embeddings])



# from transformers.models.clip.modeling_clip import CLIPTextModel, add_start_docstrings, BaseModelOutputWithPooling, CLIPTextConfig
# from typing import Any, Optional, Tuple, Union
# @add_start_docstrings(
#     """The repeated text model from CLIP without any head or projection on top."""
# )



# tokenizer = BertTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# pipeline = StableDiffusionPipeline.from_pretrained(args.model_id)
# pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
# pipeline.text_encoder = text_encoder
# # pipeline.tokenizer = tokenizer
# pipeline.to("cuda")

# images = pipeline(
#     prompt="Prompt",
#     height=512,
#     width=512,
#     num_inference_steps=25,
#     guidance_scale=4,
#     negative_prompt=None,
#     num_images_per_prompt=1,
# ).images