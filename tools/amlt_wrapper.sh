#!/bin/bash

export OUTPUT_DIR=${AMLT_OUTPUT_DIR}
export DATA_DIR=${AMLT_DATA_DIR}
if [ -n "${AMLT_MAP_INPUT_DIR}" ]; then
    export MODEL_DIR=${AMLT_MAP_INPUT_DIR}/ckpts
else
    export MODEL_DIR=${AMLT_DATA_DIR}/models
fi
export AMLT_MODE=1

bash tools/run_wrapper.sh