#!/bin/bash

export HF_HOME=/data/xxx/Downloads/huggingface
export MODEL_PATH=${HF_HOME}/models/Meta-Llama-3-8B
export MODEL_NAME=Meta-Llama-3-8B

export ENABLE_PREDICTOR=0

cd quantization

CUDA_VISIBLE_DEVICES=0 python autoclip.py \
    --model_path ${MODEL_PATH} \
    --calib_dataset pile \
    --quant_type int --w_bit 2 --q_group_size 128 \
    --run_clip --dump_clip ./clip_cache/${MODEL_NAME}/int2-g128.pt

cd ..
