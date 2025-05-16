export MODEL_PATH=${AMLT_DATA_DIR}/models/${MODEL_NAME}

export NUM_TRAIN_EPOCHS=5

export TEMPERATURE=0.2

cd train

if [ "$MODEL_NAME" = "Meta-Llama-3-8B" ] || [ "$MODEL_NAME" = "Llama-2-7b-chat-hf" ] || [ "$MODEL_NAME" = "Llama-2-13b-chat-hf" ]; then
    DATASET="mix_alpaca_c4_9000.json"
else
    DATASET="mix_wikitext_alpaca_c4_15000.json"
fi

bash train_amlt.sh \
    ${AMLT_DATA_DIR}/datasets/${MODEL_NAME}/${DATASET} \
    ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/ \
    ${AMLT_OUTPUT_DIR}/logs/${MODEL_NAME}/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME}

cd ..

export AMLT_MAP_INPUT_DIR=${AMLT_OUTPUT_DIR}

bash tools/test_task_amlt.sh