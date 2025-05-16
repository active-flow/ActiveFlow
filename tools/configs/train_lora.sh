#!/bin/bash

cd train

DATA_PATH=${DATA_DIR}/datasets/${MODEL_NAME}/${DATASET} 
SAVE_PATH=${OUTPUT_DIR}/ckpts/${MODEL_NAME}
LOGGING_DIR=${OUTPUT_DIR}/logs/${MODEL_NAME}

export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true

deepspeed --hostfile=hostfile_local --no_ssh --node_rank=0 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train.py \
    --deepspeed config/zero.json \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --threshold_path ${THRESHOLD_PATH} \
    --output_dir ${SAVE_PATH} \
    --logging_dir ${LOGGING_DIR} \
    --seed 42 \
    --num_train_epochs 2 \
    --model_max_length 512 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing False \
    --load_best_model_at_end False \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --weight_decay 0. \
    --logging_steps 1 \
    --learning_rate 0.00002 \
    --report_to "none" \
    --quant ${QUANT} \
    --bits 4 \
    --quant_type Q4_0 \
    --q_group_size 32 \
    --train_kd False \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --evaluation_strategy "epoch" \
    --use_lora ${USE_LORA} \
    --load_checkpoint False

cd ..
