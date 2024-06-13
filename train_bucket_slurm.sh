#!/bin/bash
echo COUNT_NODE=$COUNT_NODE
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
echo SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip().split('.')[0]]"`
echo THEID=$THEID

# if [[ $H == $MASTER_ADDR* ]]
# then 
# THEID=0
# else
# THEID=1
# fi
# echo THEID=$THEID

TIME_STAMP=`date "+%Y%m%d-%H%M"`
LOG_FILE="logs/${TIME_STAMP}.log"
echo $LOG_FILE
CKPT_PATH="/storage/home/lanzhenzhongLab/liling/models/stable-diffusion/sd-v1-4-full-ema.ckpt"
CKPT_PATH="/storage/home/lanzhenzhongLab/liling/models/stable-diffusion/v1-5-pruned.ckpt"
# CKPT_PATH="logs/2022-11-15T23-50-14_laionart-8gpu/checkpoints/last.ckpt"
# CONFIG_PATH="configs/based/midjourney-8gpu.yaml"
# CONFIG_PATH="configs/based/onepiece-8gpu_origin.yaml"
#SBATCH -w gvna08
#SBATCH --gres=gpu:8
#SBATCH --ntasks=32
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/20221115-1750.log
CONFIG_PATH="configs/based/laionart-8gpu-test.yaml"

# srun python main_bucket.py \
#     -t \
#     --base $CONFIG_PATH \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --scale_lr False \
#     --num_nodes 2 \
#     --check_val_every_n_epoch 1 \
#     --finetune_from $CKPT_PATH
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --nnodes=$COUNT_NODE --node_rank=$THEID  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main_bucket.py \
    -t \
    --base $CONFIG_PATH \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes $COUNT_NODE \
    --check_val_every_n_epoch 1 \
    --finetune_from $CKPT_PATH
# nohup torchrun  --nproc_per_node=8 --nnodes=4 --node_rank=3 --master_addr="172.168.125.11" --master_port=12346 main_bucket.py \
#     -t \
#     --base $CONFIG_PATH \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --scale_lr False \
#     --num_nodes 4 \
#     --check_val_every_n_epoch 1 \
#     --finetune_from $CKPT_PATH > $LOG_FILE &
# nohup torchrun  --nproc_per_node=8 --nnodes=1 main_bucket.py \
#     -t \
#     --base $CONFIG_PATH \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --scale_lr False \
#     --num_nodes 1 \
#     --check_val_every_n_epoch 1 \
#     --resume $CKPT_PATH > $LOG_FILE &
# nohup python main_bucket.py \
#     -t \
#     --base $CONFIG_PATH \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --scale_lr False \
#     --num_nodes 1 \
#     --check_val_every_n_epoch 1 \
#     --finetune_from $CKPT_PATH > $LOG_FILE &
