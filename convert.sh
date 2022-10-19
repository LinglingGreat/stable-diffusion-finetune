#!/bin/bash
CHECK_PATH="logs/2022-10-11T19-30-10_pokemon/checkpoints/epoch=000114.ckpt"
CONFIG="logs/2022-10-11T19-30-10_pokemon/configs/2022-10-11T19-30-10-project.yaml"
SAVE_PATH="logs/2022-10-11T19-30-10_pokemon/transmodel"
python scripts/convert_sd_to_diffusers.py --checkpoint_path $CHECK_PATH --original_config_file $CONFIG --dump_path $SAVE_PATH --use_ema 

