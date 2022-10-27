#!/bin/bash
#SBATCH -J llsdtrain
#SBATCH --gres=gpu:8
##SBATCH --chdir=/storage/home/lanzhenzhongLab/liling/printidea/stable-diffusion

TIME_STAMP=`date "+%Y%m%d-%H%M"`
LOG_FILE="logs/${TIME_STAMP}.log"
CKPT_PATH="/ssdwork/liling/models/stable-diffusion/sd-v1-4-full-ema.ckpt"
# CONFIG_PATH="configs/based/midjourney-8gpu.yaml"
# CONFIG_PATH="configs/based/onepiece-8gpu_origin.yaml"
CONFIG_PATH="configs/based/laionart-8gpu.yaml"
nohup python main_bucket.py \
    -t \
    --base $CONFIG_PATH \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from $CKPT_PATH > $LOG_FILE &
