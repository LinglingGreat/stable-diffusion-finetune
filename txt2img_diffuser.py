import torch
from torch import autocast
# from diffusers import StableDiffusionPipeline, DDIMScheduler
from pipeline import StableDiffusionControlSafetyCheckPipeline as StableDiffusionPipeline
import os
import pandas as pd

def image_gen(pipe, prompt, postfix, output_path, height, width, num_samples=1):
    for i in range(num_samples):
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=7.5, height=height, width=width)["sample"][0]  
        image.save(os.path.join(output_path, f"{prompt[:200]}{postfix}_{height}x{width}{str(i)}.png"))

device = "cuda"

model_id="logs/2022-10-21T17-14-25_midjourney-8gpu/train192"
output_path = "outputs/generated_mj_192"
postfix = "_576x1024"

model_id="logs/2022-10-20T16-02-47_onepiece-8gpu/train16"
output_path = "outputs/generated_op_16"
postfix = "_576x1024"

# model_id = "logs/2022-10-19T19-49-27_onepiece-8gpu/train184"
# output_path = "outputs/generated_op_184"
# postfix = "_576x1024_sd"

model_id="logs/2022-10-24T14-24-10_laionart-8gpu/train0"
output_path = "outputs/generated_laion_0"
postfix = ""

model_id="logs/2022-10-26T17-15-17_laionart-8gpu/trans/epoch=000000"
output_path = "outputs/generated_laion_1026_0"
postfix = ""

# model_id="/ssdwork/image_gen/stable-diffusion-v1-4"
# output_path = "outputs/generated_sd"
# postfix = ""

os.makedirs(output_path, exist_ok=True)

# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, use_auth_token=True
# )
# pipe = pipe.to(device)

prompt = "a beautiful girl"
# prompt = "a dog"
prompt = "beautiful rapunzel, wedding dress, beautiful face, intricate, highly detailed, 8k, textured, sharp focus, art by artgerm and greg rutkowski and alphonse mucha"

# data = pd.read_excel("测试.xlsx", sheet_name="pokemon测试", header=None)
# prompts = data.loc[:, 0].values
# print(prompts[:5])
# for prompt in prompts:

#     image_gen(prompt, postfix, height=576, width=1024, num_samples=1)
#     image_gen(prompt, postfix, height=1024, width=576, num_samples=1)

def multi_model_test():
    ckpt_path = "logs/2022-10-27T19-54-24_laionart-8gpu/trans"
    output_path = "outputs/generated_laion_10271954"
    os.makedirs(output_path, exist_ok=True)
    model_list = os.listdir(ckpt_path)

    for model_name in model_list:
        postfix = model_name
        model_id = os.path.join(ckpt_path, model_name)
        pipe = StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token=True
        )
        pipe = pipe.to(device)

        prompt = "a beautiful girl"
        # prompt = "a dog"
        prompt = "a beautiful rapunzel, wedding dress, beautiful face, intricate, highly detailed, 8k, textured, sharp focus, art by artgerm and greg rutkowski and alphonse mucha"

        image_gen(pipe, prompt, postfix, output_path, height=576, width=1024, num_samples=4)
        image_gen(pipe, prompt, postfix, output_path, height=1024, width=576, num_samples=4)
        # image_gen(prompt, postfix, height=512, width=512, num_samples=4)

multi_model_test()
