#!/bin/bash
TIME_STAMP=`date "+%Y%m%d-%H%M"`
LOG_FILE="logs/${TIME_STAMP}.log"
CKPT_PATH="/ssdwork/liling/models/stable-diffusion/sd-v1-4-full-ema.ckpt"
CKPT_PATH="logs/2022-10-12T17-16-16_onepiece-8gpu/checkpoints/last.ckpt"
nohup python main.py \
    -t \
    --base configs/stable-diffusion/onepiece-8gpu.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 2 \
    --finetune_from $CKPT_PATH > $LOG_FILE &
