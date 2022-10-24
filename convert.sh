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
python convert_sd_to_diffusers.py --checkpoint_path $CHECK_PATH --original_config_file $CONFIG --dump_path $SAVE_PATH --use_ema 

