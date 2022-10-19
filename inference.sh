# python scripts/txt2img.py \
#     --prompt 'robotic cat with wings' \
#     --outdir 'outputs/generated_pokemon' \
#     --H 512 --W 512 \
#     --n_samples 4 \
#     --config 'configs/stable-diffusion/pokemon.yaml' \
#     --ckpt 'logs/2022-10-11T19-30-10_pokemon/checkpoints/epoch=000114.ckpt'

python scripts/txt2img.py \
    --prompt 'a beautiful dressed girl, anime style' \
    --outdir 'outputs/generated_onepiece' \
    --H 512 --W 512 \
    --n_samples 4 \
    --config 'configs/stable-diffusion/onepiece-8gpu.yaml' \
    --ckpt 'logs/2022-10-13T09-29-48_onepiece-8gpu/checkpoints/epoch=000063.ckpt'
