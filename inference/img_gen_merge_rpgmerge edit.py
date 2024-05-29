import torch
import json
import csv
import os
from tqdm import tqdm
import argparse
from rpg_models.RegionalDiffusion_base import RegionalDiffusionPipeline
from rpg_models.RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default="stabilityai/stable-diffusion-2")
parser.add_argument('--prompt_path', type=str, default="datasets/ShareGPT4V/data/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json")
parser.add_argument('--output_path', type=str, default="longtext_imggen")
parser.add_argument('--cache_dir', type=str, default="/home/compu/JinProjects/jinprojects/SELMA/pretrained_models")
parser.add_argument('--eval_benchmark', type=str, default="DSG")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--text_enc', type=str, default="clip")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# model_id = "stabilityai/stable-diffusion-2"
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# model_id = "CompVis/stable-diffusion-v1-4"
model_id = args.model_id
output_path = args.output_path
batch_size = args.batch_size


# if "xl" in model_id:
#     pipe = StableDiffusionXLPipeline.from_pretrained(
#         model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir, safety_checker = None
#     )
# else:
#     pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir, safety_checker = None)

if "xl" in model_id:
    pipe = RegionalDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        safety_checker = None,
        variant="fp16",
    )
else:
    pipe = RegionalDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        safety_checker = None,
        variant="fp16",
    )

pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)
pipe.enable_xformers_memory_efficient_attention()

if args.text_enc == "longclip":
    pipe.text_encoder


## User input
# prompt = "This image provides an aerial view of an airport. The most prominent object in the photo is a Japan Airlines airplane. The airplane is white with a red tail and black lettering, standing out against the gray runway on which it is located. \n\nThe background of the image is a complex network of buildings and roads, indicative of the bustling activity typical of an airport. Among these structures, there are white storage tanks adding to the industrial feel of the scene.\n\nThe airport is surrounded by a lush expanse of green trees, providing a stark contrast to the concrete and metal structures of the airport. Despite the man-made structures, nature asserts its presence in this scene.\n\nThe photo appears to have been taken during the day under clear skies, adding to the overall clarity and detail of the image. The positioning and perspective suggest it was taken from a significant height, perhaps from another aircraft or a control tower.\n\nPlease note that this description is based on the provided image and does not include any inferred or speculative information."6
prompt = "A green twintail hair girl wearing a white shirt printed with green apple and wearing a black skirt. "
# para_dict = GPT4(prompt, key="")
split_ratio = "1;1;1"
regional_prompt = "Lush green twintails cascade down, framing the girl's face with lively eyes and a subtle smile, accented by a few playful freckles BREAK A vibrant red blouse, featuring ruffled sleeves and a cinched waist, adorned with delicate pearl buttons, radiates elegance BREAK pleated blue skirt, knee-length, sways gracefully with each step, its fabric catching the light, paired with a slender white belt for a touch of sophistication."
negative_prompt = ""
images = pipe(
    prompt=regional_prompt,
    split_ratio=None,  # The ratio of the regional prompt, the number of prompts is the same as the number of regions, and the number of prompts is the same as the number of regions
    batch_size=1,  # batch size
    base_ratio=1,  # The ratio of the base prompt
    base_prompt=prompt,
    negative_prompt=negative_prompt,  # negative prompt
    seed=2468,  # random seed
    guidance_scale=7.0,
).images[0]
images.save("test.png")

############

data_all = dict()

with open(args.prompt_path) as f:
    data = json.load(f)
f.close()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)




##############

prompts = []
names = []
sample = 'test'
for i, item in enumerate(tqdm(data)):
    prompt = item
    name = f"sd-sample{sample}-{i}"
    prompts.append(prompt)
    names.append(name)

    if len(prompts) == batch_size:
        images = pipe(prompts).images
        print(len(images))
        for j, img in enumerate(images):
            img.save(os.path.join(output_path, f"{names[j]}.jpg" ))

        prompts = []
        names = []


#############

if args.eval_benchmark == "DSG":
    data_all = dict()
    with open("../DSG/dsg-1k-anns.csv") as f:
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
