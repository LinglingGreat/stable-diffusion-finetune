import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os

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

# model_id="/ssdwork/image_gen/stable-diffusion-v1-4"
# postfix = "_sd"

os.makedirs(output_path, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, use_auth_token=True
)
pipe = pipe.to(device)

prompt = "a beautiful girl"
# prompt = "a dog"
prompt = "beautiful rapunzel, wedding dress, beautiful face, intricate, highly detailed, 8k, textured, sharp focus, art by artgerm and greg rutkowski and alphonse mucha"

for i in range(1):
    height=576
    width=1024
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, height=height, width=width)["sample"][0]  
        
    image.save(os.path.join(output_path, f"{prompt}{postfix}_{height}x{width}{str(i)}.png"))

    height=1024
    width=576
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, height=height, width=width)["sample"][0]  
        
    image.save(os.path.join(output_path, f"{prompt}{postfix}_{height}x{width}{str(i)}.png"))
