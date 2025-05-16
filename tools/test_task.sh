#!/bin/bash
# Test task script for activeflow

python train/test_task.py \
    --model=${MODEL_PATH} \
    --task=${TEST_TASK} \
    --sparse=${SPARSE} \
    --do_cr=${DO_CR} \
    --threshold_path=${THRESHOLD_PATH} \
    --sparse_strategy=${SPARSE_STRATEGY} \
    --test_all=${TEST_ALL} \
    --seed=42 \
    --limit=${LIMIT:--1} \
    --num_shot=5 \
    --batch_size=1 \
    --quant=${QUANT} \
    --use_lora=${USE_LORA} \
    --lora_checkpoint=${LORA_CHECKPOINT}