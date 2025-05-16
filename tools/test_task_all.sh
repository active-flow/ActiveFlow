#!/bin/bash
CHECKPOINT_DIRS=$(find $MODEL_PATH -maxdepth 1 -type d -name "checkpoint-*")

for CHECKPOINT_DIR in $CHECKPOINT_DIRS; do
    echo "Processing checkpoint directory: $CHECKPOINT_DIR"
    
    MODEL_PATH=$CHECKPOINT_DIR
    echo "CHECKPOINT_PATHCHECKPOINT_PATH: $MODEL_PATH"
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
        --quant=1
    
    echo "Finished processing checkpoint directory: $CHECKPOINT_DIR"
done