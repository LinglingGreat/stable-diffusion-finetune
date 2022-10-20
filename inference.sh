# python scripts/txt2img.py \
#     --prompt 'robotic cat with wings' \
#     --outdir 'outputs/generated_pokemon' \
#     --H 512 --W 512 \
#     --n_samples 4 \
#     --config 'configs/stable-diffusion/pokemon.yaml' \
#     --ckpt 'logs/2022-10-11T19-30-10_pokemon/checkpoints/epoch=000114.ckpt'

outputdir='outputs/generated_onepiece'
config='configs/stable-diffusion/onepiece-8gpu.yaml'
ckpt='logs/2022-10-13T09-29-48_onepiece-8gpu/checkpoints/epoch=000063.ckpt'

outputdir='outputs/generated_onepiece_bucket'
config='configs/based/onepiece-8gpu.yaml'
ckpt='logs/2022-10-20T16-02-47_onepiece-8gpu/checkpoints/epoch=000003.ckpt'

python scripts/txt2img.py \
    --prompt 'a beautiful dressed girl, anime style' \
    --outdir $outputdir \
    --H 512 --W 1024 \
    --n_samples 4 \
    --config $config \
    --ckpt $ckpt
