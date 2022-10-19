import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os

model_id="/ssdwork/liling/stable-diffusion/logs/2022-10-11T19-30-10_pokemon/transmodel"
model_id="/ssdwork/liling/stable-diffusion/logs/2022-10-11T19-30-10_pokemon/transmodelnoema"
device = "cuda"
output_path = "outputs/generated_pokemon_diffusers"
postfix = "noema"
os.makedirs(output_path, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, use_auth_token=True
)
pipe = pipe.to(device)

prompt = "robotic cat with wings"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save(os.path.join(output_path, prompt+postfix+".png"))
