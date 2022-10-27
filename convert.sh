#!/bin/bash
CHECK_PATH="logs/2022-10-20T16-02-47_onepiece-8gpu/checkpoints/epoch=000016.ckpt"
CONFIG="logs/2022-10-20T16-02-47_onepiece-8gpu/configs/2022-10-20T16-02-47-project.yaml"
SAVE_PATH="logs/2022-10-20T16-02-47_onepiece-8gpu/train16"

CHECK_PATH="logs/2022-10-21T17-14-25_midjourney-8gpu/checkpoints/epoch=000192.ckpt"
CONFIG="logs/2022-10-21T17-14-25_midjourney-8gpu/configs/2022-10-21T17-14-25-project.yaml"
SAVE_PATH="logs/2022-10-21T17-14-25_midjourney-8gpu/train192"

CHECK_PATH="logs/2022-10-19T19-49-27_onepiece-8gpu/checkpoints/epoch=000184.ckpt"
CONFIG="logs/2022-10-19T19-49-27_onepiece-8gpu/configs/2022-10-19T19-49-27-project.yaml"
SAVE_PATH="logs/2022-10-19T19-49-27_onepiece-8gpu/train184"

CHECK_PATH="logs/2022-10-24T14-24-10_laionart-8gpu/checkpoints/epoch=000000.ckpt"
CONFIG="logs/2022-10-24T14-24-10_laionart-8gpu/configs/2022-10-24T14-24-10-project.yaml"
SAVE_PATH="logs/2022-10-24T14-24-10_laionart-8gpu/train0"
# python convert_sd_to_diffusers.py --checkpoint_path $CHECK_PATH --original_config_file $CONFIG --dump_path $SAVE_PATH --use_ema 

ckpt_path="logs/2022-10-26T17-15-17_laionart-8gpu/checkpoints"
CONFIG="logs/2022-10-26T17-15-17_laionart-8gpu/configs/2022-10-26T17-15-17-project.yaml"

for CHECK_PATH in `find $ckpt_path -type f`
do  
SAVE_PATH=${CHECK_PATH/checkpoints/trans}
SAVE_PATH=${SAVE_PATH/.ckpt/}
echo $CHECK_PATH
echo $SAVE_PATH
if [ -d "$SAVE_PATH" ];
    then echo "Already Exists!"
else
    python convert_sd_to_diffusers.py --checkpoint_path $CHECK_PATH --original_config_file $CONFIG --dump_path $SAVE_PATH --use_ema 
fi
done