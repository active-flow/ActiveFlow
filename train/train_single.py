import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from datasets import load_dataset
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import logging
import transformers
from transformers import HfArgumentParser, TrainingArguments
from utils.sparse_hook import prepare_sparse_hook
from quantization.qlinear import QLinear, convertModelToQuant
from quantization.clip_utils import apply_clip
import logging
from dataclasses import dataclass, field
from train.mytrainer_new import KDTrainer
import random
from tqdm import tqdm
from datasets import load_dataset

from utils.sparse_hook import prepare_sparse_hook, get_sparsity_configs
# 如果需要使用LoRA，则导入PEFT相关模块
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 定义模型参数类
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/xxx/models/Qwen2.5-0.5B-Instruct")

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
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for fine-tuning."})
def preprocess_wiki(examples, tokenizer):
    # 获取批次中的文本列表
    texts = examples["text"]
    # 对文本进行 tokenization，填充到固定长度，并截断过长文本
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # 返回字典，其中 labels 直接复制 input_ids
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()
    }

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
        self.data = self.data.map(lambda x: preprocess_wiki(x, tokenizer), batched=True)
        self.input_ids = torch.tensor(self.data["input_ids"])
        self.attention_mask = torch.tensor(self.data["attention_mask"])
        self.labels = torch.tensor(self.data["labels"])


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# 定义训练函数
def train():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)
    # 设置随机种子
    random.seed(training_args.seed)

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    print(model)
    prepare_sparse_hook(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # load quantization if specified
    if True or training_args.quant_type is not None:
        print("converting the model to qat, this may take a while...")
        model, _ = convertModelToQuant(model, compute_dtype=torch.bfloat16, quant_type='Q4_0', q_group_size=64)
        
    # 准备模型进行LoRA训练
    if training_args.use_lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 根据模型类型调整
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 加载数据集
    dataset = CustomDataset(data_path="/home/donglinbai/Projects/xxx/BitDistiller-Q4_0/data/datasets/Qwen2.5-0.5B-Instruct/mix_wikitext_alpaca_c4_12.json", tokenizer=tokenizer)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    # 定义训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()