#!/bin/bash

export PATH=/home/aiscuser/.local/bin:${PATH}

prepareSSHKey() {
    cp ssh/* ~/.ssh
}

prepareSSHKey

pip uninstall -y onnxruntime_training

pip install -r requirements.txt

mkdir -p 3rdparty

cd 3rdparty

installLMEval() {
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e .
    cd ..
}

prepare() {
    wandb offline
}

installLMEval

prepare
