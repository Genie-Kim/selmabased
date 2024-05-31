import torch
from diffusers.schedulers import DPMSolverMultistepScheduler, KarrasDiffusionSchedulers

from mllm import GPT4, local_llm
from RegionalDiffusion_base import RegionalDiffusionPipeline
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline

# If you want to use load ckpt, initialize with ".from_single_file".
# pipe = RegionalDiffusionXLPipeline.from_single_file("../models/anything-v3", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# If you want to use diffusers, initialize with ".from_pretrained".
pipe = RegionalDiffusionXLPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)
pipe.enable_xformers_memory_efficient_attention()
## User input
# prompt = "This image provides an aerial view of an airport. The most prominent object in the photo is a Japan Airlines airplane. The airplane is white with a red tail and black lettering, standing out against the gray runway on which it is located. \n\nThe background of the image is a complex network of buildings and roads, indicative of the bustling activity typical of an airport. Among these structures, there are white storage tanks adding to the industrial feel of the scene.\n\nThe airport is surrounded by a lush expanse of green trees, providing a stark contrast to the concrete and metal structures of the airport. Despite the man-made structures, nature asserts its presence in this scene.\n\nThe photo appears to have been taken during the day under clear skies, adding to the overall clarity and detail of the image. The positioning and perspective suggest it was taken from a significant height, perhaps from another aircraft or a control tower.\n\nPlease note that this description is based on the provided image and does not include any inferred or speculative information."6
prompt = "A green twintail hair girl wearing a white shirt printed with green apple and wearing a black skirt."
para_dict = GPT4(prompt, key="")
split_ratio = para_dict["Final split ratio"]
regional_prompt = para_dict["Regional Prompt"]
negative_prompt = ""
images = pipe(
    prompt=regional_prompt,
    split_ratio=split_ratio,  # The ratio of the regional prompt, the number of prompts is the same as the number of regions, and the number of prompts is the same as the number of regions
    batch_size=1,  # batch size
    base_ratio=0.5,  # The ratio of the base prompt
    base_prompt=prompt,
    num_inference_steps=20,  # sampling step
    height=1024,
    negative_prompt=negative_prompt,  # negative prompt
    width=1024,
    seed=2468,  # random seed
    guidance_scale=7.0,
).images[0]
images.save("test.png")
