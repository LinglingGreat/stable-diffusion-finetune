#!/bin/bash
#SBATCH --job-name=sdtrain3     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --output=slurm/2nodes.out
#SBATCH -w gvna17        # specified nodes

# sent to sub script
# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_PORT=$master_port
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# ******************************************************************************************
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "MASTER_PORT:= " $MASTER_PORT
echo "WORLD_SIZE:= " $WORLD_SIZE
echo "MASTER_ADDR:= " $MASTER_ADDR
echo "COUNT_NODE:= " $COUNT_NODE
echo "HOSTNAMES:= " $HOSTNAMES
echo "GPUS:= " $SLURM_GPUS_ON_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

srun train_bucket_slurm.sh

