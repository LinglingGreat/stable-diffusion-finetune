#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

outputdir='outputs/generated_onepiece'
config='configs/stable-diffusion/onepiece-8gpu.yaml'
ckpt='logs/2022-10-13T09-29-48_onepiece-8gpu/checkpoints/epoch=000063.ckpt'

outputdir='outputs/generated_midjourney_bucket_916'
config='configs/based/onepiece-8gpu.yaml'
ckpt='logs/2022-10-20T16-02-47_onepiece-8gpu/checkpoints/epoch=000010.ckpt'
ckpt="/ssdwork/liling/models/stable-diffusion/sd-v1-4-full-ema.ckpt"
# ckpt='logs/2022-10-19T19-49-27_onepiece-8gpu/checkpoints/epoch=000184.ckpt'

ckpt='logs/2022-10-24T14-24-10_laionart-8gpu/checkpoints/epoch=000000-v1.ckpt'
outputdir='outputs/generated_laionart'
config='configs/based/laionart-8gpu.yaml'

# python txt2img.py \
#     --prompt 'a beautiful girl' \
#     --outdir $outputdir \
#     --H 576 --W 1024 \
#     --n_samples 1 \
#     --config $config \
#     --ckpt $ckpt

outputdir='outputs/generated_laionart'
config='configs/based/laionart-8gpu.yaml'
ckpt_path="logs/2022-10-24T14-24-10_laionart-8gpu/checkpoints"
for ckpt in `find $ckpt_path -type f`
do  
echo $ckpt
python txt2img.py \
    --prompt 'a beautiful girl' \
    --outdir $outputdir \
    --H 576 --W 1024 \
    --n_samples 4 \
    --config $config \
    --ckpt $ckpt
done 