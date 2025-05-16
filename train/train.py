import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from quantization.qlinear import QLinear, convertModelToQuant
from quantization.clip_utils import apply_clip

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, BitsAndBytesConfig, default_data_collator
from datasets import load_dataset
import json
import glob
import torch.distributed as dist

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from train.mytrainer import KDTrainer
from train.utils import make_supervised_data_module, smart_tokenizer_and_embedding_resize
import random
from tqdm import tqdm
from datasets import load_dataset

from utils.sparse_hook import prepare_sparse_hook, get_sparsity_configs
# 如果需要使用LoRA，则导入PEFT相关模块
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training

def get_last_checkpoint(output_dir: str) -> Optional[str]:
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda ckpt: int(ckpt.split("-")[-1]))
    return checkpoints[-1]

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    threshold_path: str = field(default=None, metadata={"help": "Path to save threshold"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."},
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    logging_dir: str = field(default="logs")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=False)
    bits: int = field(default=2, metadata={"help": "How many bits to use."})
    q_group_size: int = field(default=128, metadata={"help": "Quantization Group Size."})
    quant_type: str = field(
        default="int2-asym",
        metadata={"help": "Quantization data type to use. Should be one of `int2-asym` or `ste-n2f3`."}
    )
    clip: str = field(default=None, metadata={"help": "The path of clip cache"})
    train_kd: bool = field(default=False, metadata={"help": "Whether to use KD to QAT"})
    kd_tmp: int = field(default=1, metadata={"help": "Temperature of KD"})
    kd_loss_type: str = field(default=None, metadata={"help": "Type of loss function when KD-QAT"})
    cakld_steps: int = field(default=10, metadata={"help": "How many steps to calculate the coefficient of CAKLD."})
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy to adopt during training. Options: 'no', 'steps', 'epoch'."}
    )
    eval_steps: int = field(default=500, metadata={"help": "Number of update steps between two evaluations."})
    quant : int= field(default=0, metadata={"help": "Whether to use quantization."})
    use_lora: int = field(default=0, metadata={"help": "Whether to use LoRA for fine-tuning."})
    load_checkpoint: bool = field(default=False, metadata={"help": "Whether to load checkpoint."})

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    device_map = "auto"
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    print("device_map", device_map)
    print(f"loading {model_args.model_name_or_path} model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    # model.config.use_cache = False
    # prepare sparse
    prepare_sparse_hook(model)
    global_weight_preditor = model.predictor
    if global_weight_preditor is not None:
        attn_sp, mlp_sp, w_p, do_cr = get_sparsity_configs()
        global_weight_preditor.set_sp_config(attn_sp, mlp_sp, w_p)
        global_weight_preditor.set_do_pre_prediction(do_cr)
        global_weight_preditor.set_sparsity_threshold(data_args.threshold_path)
        print('sparse: ', attn_sp, mlp_sp , w_p)

    print('use_lora', training_args.use_lora)
    if training_args.use_lora:
        print("Applying LoRA fine-tuning...")
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # target_modules=["q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    pad_status = True
    if tokenizer.pad_token is None:
        print("tokenizer has not padding token")
        pad_status = False
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # load quantization if specified
    if training_args.quant:
        print('begin quantization')
        if training_args.quant_type is not None:
            print("converting the model to qat, this may take a while...")
            model, _ = convertModelToQuant(model, compute_dtype=torch.bfloat16, quant_type=training_args.quant_type, q_group_size=training_args.q_group_size)
        if training_args.clip is not None:
            q_config = {"zero_point": True, "q_group_size": training_args.q_group_size}
            print("Loading pre-computed Clipping results from", training_args.clip)
            clip_results = torch.load(training_args.clip)
            apply_clip(model, clip_results)
            print("Clipping init successfully!")
    else :
        print("No quantization is used, using the original model")
    
    # load teacher model if KD training is enabled (assume LoRA and KD are mutually exclusive)
    if training_args.train_kd:
        print("loading Teacher Model...")
        # TODO: teacher model for loading?
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_4bit=False,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            # max_memory=max_memory,
        )
        teacher_model.eval()
        teacher_model.cuda()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        if pad_status is False:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=teacher_model,
            )
        model.kd_loss_scale = 1.0
        print("Teacher Model loaded")

    # compute cakld coefficient if needed
    mean_prob = 0
    if training_args.train_kd and training_args.kd_loss_type == "cakld":
        print("Get the main Prob!")
        probDataloader = DataLoader(
            data_module['train_dataset'], 
            shuffle=True, 
            collate_fn=data_module['data_collator'], 
            batch_size=training_args.per_device_train_batch_size,
            drop_last=True,
        )
        prob = 0
        for step, batch in tqdm(enumerate(probDataloader)):
            if step > training_args.cakld_steps:
                break
            batch = {k: v.to(teacher_model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = teacher_model(**batch)
            logits = outputs.get("logits").contiguous()
            prob1 = torch.nn.functional.softmax(logits, dim=-1)
            prob1 = torch.max(prob1, dim=-1).values 
            prob += prob1.mean()
        mean_prob = prob / training_args.cakld_steps
        mean_prob = torch.Tensor(mean_prob.to(teacher_model.device))
        dist.all_reduce(mean_prob, op=dist.ReduceOp.SUM)
        mean_prob = mean_prob / dist.get_world_size()
        print(f"Get the coefficient: {mean_prob}")
    model.to('cpu')
    # load trainer (这里假设LoRA和KD微调互斥)
    if training_args.train_kd:
        trainer = KDTrainer(model=model, tokenizer=tokenizer, teacher_model=teacher_model, loss_type=training_args.kd_loss_type, mean_prob=mean_prob, args=training_args, **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, "requires grad")
    #     else :
    #         print(name, "None requires grad")
    # for param in model.parameters():
    #     param.requires_grad = True
    print('begin training')
    print('load_checkpoint', training_args.load_checkpoint)
    if training_args.load_checkpoint :
        resume_ckpt = get_last_checkpoint(training_args.output_dir)
        print('resume_ckpt', resume_ckpt)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else :
        trainer.train()
    print('finish training')
    if training_args.use_lora:
        print("Saving LoRA weights...")
        model.save_pretrained(training_args.output_dir)
    else :
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()