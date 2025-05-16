#!/bin/bash

#--limit=500
python train/test_chat.py \
    --model=${MODEL_PATH} \
    --seed=42 \
    --sparse=${SPARSE} \
    --do_cr=${DO_CR} \
    --threshold_path=${THRESHOLD_PATH} \
    --sparse_strategy=${SPARSE_STRATEGY} \
    --batch_size=1 \
    --quant=0 
