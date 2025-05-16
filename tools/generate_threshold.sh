#!/bin/bash

python train/threshold_generation.py \
    --model=${MODEL_PATH} \
    --seed=42 \
    --task=${TEST_TASK} \
    --limit=100 \
    --num_shot=0 \
    --file_path=${OUTPUT_DIR}/threshold/${MODEL_NAME}-all \
    --sparse=0 \
    --do_generation \
    --save_activations \
    --test_all=1
