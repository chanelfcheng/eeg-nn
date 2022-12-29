#!/bin/sh

export GPU_ID=$1

echo $GPU_ID

# cd ..
export DATASET_DIR="Data"
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python test.py --name_of_args_json_file exp_config/seed-v_maml++.json --gpu_to_use $GPU_ID
# python train_maml_system.py --name_of_args_json_file exp_config/seed-v_maml++.json --gpu_to_use $GPU_ID