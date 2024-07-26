from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.schedulers import (
    KarrasDiffusionSchedulers,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
import torch
import json
import os
from tqdm import tqdm
import argparse
from models.RepeatTextencStableDiffusion_base import (
    RepeatTextencStableDiffusionPipeline,
)
from models.rpg_models.RegionalDiffusion_base import RegionalDiffusionPipeline
from models.rpg_models.RegionalDiffusion_xl import RegionalDiffusionXLPipeline
import multiprocessing
import copy

def divide_input_tuple_list(input_tuple_list, worker_num):
    k, m = divmod(len(input_tuple_list), worker_num)
    return [
        input_tuple_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(worker_num)
    ]

def pipename_listinpipename(pipename, candidates):
    for candidate in candidates:
        if candidate in pipename:
            return True
    return False

def parse_arguments():
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
            "clip_repeattext_30",
            "clip_repeattext_40",
            "clip_repeattext_50",
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

    parser.add_argument("--worker_num_pergpu", type=int, default=1)
    parser.add_argument("--gpu_num", nargs="+",type=str, default=["0","1"])
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # model_id = "stabilityai/stable-diffusion-2-1"
    # model_id = "runwayml/stable-diffusion-v1-5"

    # model_id = "stabilityai/stable-diffusion-2"
    # model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    # model_id = "CompVis/stable-diffusion-v1-4"


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.seed = 2468
    args.output_path = os.path.join(args.exp_path, args.model_id.split("/")[-1], f"{args.pipename}_output")

    sorted_args = sorted(vars(args).items())
    for arg, value in sorted_args:
        # Print argument names in blue and values in green for clear differentiation
        print(f"\033[34m{arg}\033[0m: {value}")
    input("Press Enter to continue...")
    
    return args

def generate_images(inputlist_config_data_gpunum):
    config = inputlist_config_data_gpunum[0]
    data_list = inputlist_config_data_gpunum[1]
    gpu_num = inputlist_config_data_gpunum[2]
    config['device'] = torch.device(f"cuda:{gpu_num}")
    
    # initinalization
    if pipename_listinpipename(config['pipename'], ["longclip","clip_repeattext","clip_repeattext_30", "clip_repeattext_40",
            "clip_repeattext_50","clip_normal","clip_summary"]):
        if config['pipename'] == "longclip":
            assert config['model_id'] != "stabilityai/stable-diffusion-2", "LongCLIP model is not supported SD v2.1"
            longclip_modelpath = os.path.join(config['cache_dir'], "LongCLIP/longclip-L.pt")
        else:
            longclip_modelpath = None
        
        if "clip_repeattext" in config['pipename']:
            repeat_tokenlen = int(config['pipename'].split("_")[-1])
        
        if "xl" in config['model_id']:
            # TODO: XL 구현
            print("not implemented")
        else:
            pipe = RepeatTextencStableDiffusionPipeline.from_pretrained(
                config['model_id'],
                torch_dtype=torch.float16,
                cache_dir=config['cache_dir'],
                safety_checker=None,
                repeat_tokenlen=repeat_tokenlen,
                longclip_modelfile=longclip_modelpath,
            )
            # # for match with rpg model's configuration.
            # num_inference_steps=20,  # sampling step
            # height=1024,
            # width=1024,
            # guidance_scale=7.0,


    # TODO: implement abstractdiffusion
    elif pipename_listinpipename(config['pipename'], ["abstractdiffusion"]):
        if "xl" in config['model_id']:
            print("not implemented")

        else:
            pipe = RepeatTextencStableDiffusionPipeline.from_pretrained(
                config['model_id'],
                torch_dtype=torch.float16,
                cache_dir=config['cache_dir'],
                safety_checker=None,
                repeat_tokenlen=repeat_tokenlen,
                longclip_modelfile=longclip_modelpath,
            )

    elif pipename_listinpipename(config['pipename'], ["regionaldiffusion"]):
        if "xl" in config['model_id']:
            pipe = RegionalDiffusionXLPipeline.from_single_file(
                "../models/anything-v3",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        else:
            pipe = RegionalDiffusionPipeline.from_pretrained(
                config['model_id'],
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
    pipe.to(config['device'])
            
    id2summary = {}
    with open(config['summary_path'], "r") as f:
        for line in f:
            item = json.loads(line)
            key = item.pop("example_id")
            id2summary[key] = item

    ##############
    
    prompts = []
    filenames = []
    generator = torch.cuda.manual_seed_all(config['seed'])
    data_len = len(data_list)
    
    for data_dict in tqdm(data_list):
        split = data_dict["split"]
        if split in config['splits']:
            output_foldername=os.path.join(config['output_path'],split)
            count = 0 # for batch inference
            example_id = data_dict["example_id"]
            if config['pipename'] in ["clip_summary"] and id2summary[example_id]['long_bool']:
                prompt = id2summary[example_id]['summary'] # if long, use summary
            else:
                prompt = data_dict["description"]
            filename = data_dict["image_file"]
            
            prompts.append(prompt)
            filenames.append(filename)
            
            count += 1
            if len(prompts) == config['batch_size'] or data_len - count < data_len % config['batch_size']:
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
    return True

if __name__ == "__main__":
    
    args = parse_arguments()
    
    os.makedirs(args.output_path, exist_ok=True)
    for split in args.splits:
        os.makedirs(os.path.join(args.output_path,split),exist_ok=True)
    
    id2datas={}
    with open(args.prompt_path, "r") as f:
        for line in f:
            item = json.loads(line)
            key = item["example_id"]
            split = item["split"]
            if split in args.splits:
                id2datas[key] = item
    
    
        
    if args.debug:
        generate_images((vars(args),list(id2datas.values()),"0"))
    else:
        # Process the data (example)
        print("prepareing multi-processing")               
        divided_input_list = divide_input_tuple_list(list(id2datas.keys()), args.worker_num_pergpu*len(args.gpu_num))   
        
        divided_inputdata_list = []
        for keys_perworker in divided_input_list:
            datas_perworker = []
            for key in keys_perworker:
                datas_perworker.append(id2datas.pop(key))
            divided_inputdata_list.append(datas_perworker)
        
        # append gpu number and args
        for i,gpunum in enumerate(args.gpu_num):
            for j in range(len(divided_inputdata_list)):
                if i*args.worker_num_pergpu-1<j and j<(i+1)*args.worker_num_pergpu:
                    config = {k: copy.deepcopy(v) for k, v in vars(args).items()}
                    divided_inputdata_list[j] = (config, divided_inputdata_list[j],gpunum)

        
        torch.multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(args.worker_num_pergpu*len(args.gpu_num)) as p:
            output_status = list(p.imap(generate_images, divided_inputdata_list))
        
        print("multi processing done.")
        