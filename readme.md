# ActiveFlow

## Overview

This repository implements a complete sparse inference and training framework that supports one‑click execution of Quantization‑Aware Training (QAT), self‑distillation, LoRA fine‑tuning, evaluation, and inference. GPU parallelization is natively supported.

This repository is based on Bitdistiller.

## Key Features
- **One‑Click Workflow**  
  Execute the entire pipeline—QAT, self‑distillation, LoRA fine‑tuning, evaluation, and inference—with a single command.
- **Quantization‑Aware Training (QAT)**  
  Incorporates quantization noise during training to improve the accuracy of quantized models.
- **Self‑Distillation**  
  Uses a teacher–student setup to further boost model performance through self‑distillation.
- **LoRA Fine‑Tuning**  
  Leverages Low‑Rank Adaptation to efficiently fine‑tune large models.
- **Evaluation & Inference**  
  Built‑in support for validation metrics on test datasets and deployment‑ready inference.
- **GPU Parallelism**  
  Out‑of‑the‑box multi‑GPU support to accelerate training and inference.

## Easy Setup

Deploy it with conda:

```
    conda create -n activeflow python=3.10 -y
    conda activate activeflow
    pip install -e .
```

## Quick Start

1. **Configure Environment Variables**  
   Open `tools/params_temp.env` and fill in your project paths, dataset paths, and training parameters following the examples in `tools/configs/`.

2. **Run the Pipeline**  
   ```bash
   cd tools
   bash run_wrapper.sh
   ```
    
   This script will automatically run QAT, self‑distillation, LoRA fine‑tuning, evaluation, and inference in sequence based on your params_temp.env settings.
