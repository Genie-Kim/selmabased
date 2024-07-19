from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.schedulers import (
    KarrasDiffusionSchedulers,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
from models.LongCLIP import longclip
import shutil
import torch
import json
import csv
import os
from tqdm import tqdm
import argparse
from models.RepeatTextencStableDiffusion_base import (
    RepeatTextencStableDiffusionPipeline,
)
from models.rpg_models.RegionalDiffusion_base import RegionalDiffusionPipeline
from models.rpg_models.RegionalDiffusion_xl import RegionalDiffusionXLPipeline


parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt_path",
    type=str,
    default="datasets/docci/docci_descriptions.jsonlines",
)
parser.add_argument(
    "--summary_path",
    type=str,
    default="datasets/docci/summary_docci_testqual.jsonlines"
)
parser.add_argument("--exp_path", type=str, default="exp_results")
parser.add_argument(
    "--cache_dir",
    type=str,
    default="pretrained_models",
)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument(
    "--pipename",
    type=str,
    choices=[
        "regionaldiffusion",
        "longclip",
        "clip_repeattext",
        "clip_normal",
        "clip_summary",
        "abstractdiffusion",
    ],
    default="clip_normal",
)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "runwayml/stable-diffusion-v1-5"

# model_id = "stabilityai/stable-diffusion-2"
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# model_id = "CompVis/stable-diffusion-v1-4"

model_id = args.model_id
batch_size = args.batch_size
args.seed = 2468

args.output_path = os.path.join(args.exp_path, model_id.split("/")[-1], f"{args.pipename}_output")

sorted_args = sorted(vars(args).items())
for arg, value in sorted_args:
    # Print argument names in blue and values in green for clear differentiation
    print(f"\033[34m{arg}\033[0m: {value}")
input("Press Enter to continue...")

os.makedirs(args.output_path, exist_ok=True)
for split in args.splits:
    os.makedirs(os.path.join(args.output_path,split),exist_ok=True)


if args.pipename in ["longclip","clip_repeattext","clip_normal","clip_summary"]:

    if args.pipename == "longclip":
        assert model_id != "stabilityai/stable-diffusion-2", "LongCLIP model is not supported SD v2.1"
        longclip_modelpath = os.path.join(args.cache_dir, "LongCLIP/longclip-L.pt")
    else:
        longclip_modelpath = None
    
    if args.pipename == "clip_repeattext":
        repeat_textenc = True
    else:
        repeat_textenc = False
    
    
    if "xl" in model_id:
        # TODO: XL 구현
        print("not implemented")
    else:
        pipe = RepeatTextencStableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
            safety_checker=None,
            repeat_textenc=repeat_textenc,
            longclip_modelfile=longclip_modelpath,
        )
        # # for match with rpg model's configuration.
        # num_inference_steps=20,  # sampling step
        # height=1024,
        # width=1024,
        # guidance_scale=7.0,
        
elif args.pipename in ["abstractdiffusion"]:
    if "xl" in model_id:
        print("not implemented")

    else:
        pipe = RepeatTextencStableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
            safety_checker=None,
            repeat_textenc=args.repeat_textenc,
            longclip_modelfile=longclip_modelpath,
        )
        # TODO: add seed fix option....
        # # for match with rpg model's configuration.
        # num_inference_steps=20,  # sampling step
        # height=1024,
        # width=1024,
        # guidance_scale=7.0,

elif args.pipename == "regionaldiffusion":
    if "xl" in model_id:
        pipe = RegionalDiffusionXLPipeline.from_single_file(
            "../models/anything-v3",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

    else:
        pipe = RegionalDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
else:
    raise ValueError("Invalid pipename")


# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
num_inference_steps = 50
pipe.set_progress_bar_config(disable=True)
pipe.to(device)


split2ids={}
with open(args.prompt_path, "r") as f:
    for line in f:
        item = json.loads(line)
        split = item.pop("split")
        key = item.pop("example_id")
        if split not in split2ids:
            split2ids[split] = {}
        split2ids[split][key] = item
        
id2summary = {}
with open(args.summary_path, "r") as f:
    for line in f:
        item = json.loads(line)
        key = item.pop("example_id")
        id2summary[key] = item

##############

prompts = []
filenames = []
generator = torch.cuda.manual_seed_all(args.seed)
for split, id2datas in tqdm(split2ids.items()):
    
    if split in args.splits:
        output_foldername=os.path.join(args.output_path,split)
        count = 0
        data_len = len(id2datas)
        for example_id, value in tqdm(id2datas.items()):
            if args.pipename in ["clip_summary"] and id2summary[example_id]['long_bool']:
                prompt = id2summary[example_id]['summary'] # if long, use summary
            else:
                prompt = value["description"]
            filename = value["image_file"]
            
            prompts.append(prompt)
            filenames.append(filename)
            
            count += 1
            if len(prompts) == batch_size or data_len - count < data_len % batch_size:
                images = pipe(
                    prompts,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    # height=height,
                    # width=width,
                ).images
                for j, img in enumerate(images):
                    img.save(os.path.join(output_foldername, f"{filenames[j]}"))

                prompts = []
                filenames = []

            # if count > 100:
            #     break


#############

# if args.eval_benchmark == "DSG":
#     data_all = dict()
#     with open("DSG/dsg-1k-anns.csv") as f:
#         data = csv.reader(f, delimiter=",")
#         next(data)
#         for row in data:
#             if row[0] in data_all:
#                 continue
#             data_all[row[0]] = row[1]
#     f.close()
#     print(len(data_all))

#     for k, v in tqdm(data_all.items()):
#         name = k
#         prompt = v
#         images = pipe(prompt, num_images_per_prompt=1, cross_attention_kwargs={"scale": scale / 100}).images[0]
#         images.save(os.path.join(output_path, f"{name}.jpg"))

# elif args.eval_benchmark == "TIFA":
#     with open("../TIFA/tifa_v1.0/tifa_v1.0_text_inputs.json", "r") as f:
#         data_all = json.load(f)
#     f.close()

#     for item in tqdm(data_all):
#         prompt = item["caption"]
#         name = item["id"]
#         images = pipe(prompt, num_images_per_prompt=1, cross_attention_kwargs={"scale": scale / 100}).images[0]
#         images.save(os.path.join(output_path, f"{name}.jpg"))


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
