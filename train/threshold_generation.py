import gc
import os
import argparse
import numpy as np
import fnmatch
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Function
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from utils.sparse_hook import prepare_sparse_hook
from utils.models import get_llm
from quantization.qlinear import quant_and_dequant_model_q4_0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--task', type=str, help='Task.')
    parser.add_argument('--limit', type=int, default=-1, help='Number of test samples.')
    parser.add_argument('--num_shot',type=int, default=0,help='NUM_SHOT')
    parser.add_argument('--file_path',type=str, default=0,help='file_path')
    parser.add_argument('--do_generation', action='store_true', default=False, help='Enable generation mode')
    parser.add_argument('--save_activations', action='store_true', default=False, help='Enable save activation mode')
    parser.add_argument('--sparse',type=int,default=0.5)
    parser.add_argument('--test_all',type=bool,default=0)
    args = parser.parse_args()
    return args



def eval_inference(
    model,
    tokenizer,
    task_list,
    limit=None
):
    #from .llm_eval_adaptor import LMEvalAdaptor
    from lm_eval import evaluator
    from lm_eval.utils import make_table
    from lm_eval.tasks import TaskManager
    from lm_eval.models.huggingface import HFLM
    task_manager = TaskManager()
    task_names = task_manager.match_tasks(task_list)
    batch_size = 1
    lm_eval_model = HFLM(model, tokenizer=tokenizer, batch_size=batch_size, max_length=1024)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        num_fewshot=0,
        batch_size=batch_size,
        device=None,
        use_cache=None,
        limit=limit,
        check_integrity=False,
    )

    return make_table(results)

from datasets import load_dataset

# def eval_generate(
#     model,
#     tokenizer,
#     dataset_name,
#     limit=None
# ):
#     # dataset = load_dataset(dataset_name)
#     dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
#     if limit is not None:
#         dataset = dataset['test'].shuffle(seed=42).select(range(limit))
#         batch_size = 16
#     for batch in dataset['test'].to_dict('records'):
#         input_ids = tokenizer(batch['input'], return_tensors='pt').to(model.device).input_ids
#         # 推理
#         outputs = model.generate(
#             input_ids=input_ids,
#             max_length=1024
        # )
    
def main():
    args = parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model)
    prepare_sparse_hook(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # quant_and_dequant_model_q4_0(model)

    print("task: eval")

    #task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    #task_list = ["boolq"]
    #task_name = os.environ["EVAL_DATASET"]
    
    print(f"task_name: {args.task}")
    
    limit = None if args.limit <= 0  else args.limit
    
    task_list = [args.task, 'wikitext']
    
    if args.test_all:
        sp_configs = [(0,0,0,0), (0.5,0.5,0,0), (0.6,0.6,0,0), (0.7,0.7,0,0), (0.8,0.8,0,0),]
    else :
        sp_configs = [(args.sparse, args.sparse, 0, 0)]
    

    

    def get_dataset(dataset_name, subset, split, size=None, start=0):
        if size is None:
            dataset = load_dataset(dataset_name, subset, trust_remote_code=True)[split]
        else:
            dataset = load_dataset(dataset_name, subset, streaming=True, trust_remote_code=True)[split]
            dataset = dataset.skip(start).take(size)

        return dataset

    tokenizer.pad_token = tokenizer.eos_token
    # build histograms
    dataset = get_dataset(
        "wikitext",
        subset="wikitext-2-v1",
        split="train",
        size=300,
    )
    text = ""
    for sample in tqdm(dataset):
        text += sample["text"] + "\n\n"
    
    bsz, seq_len = 10, 2048

    encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")

    input_ids = encodings.input_ids[:bsz,:].to(device="cuda:0")
    print(input_ids.size(), input_ids.shape)

    global_weight_preditor = model.predictor
    
    global_weight_preditor.set_cal_activations(args.file_path)

    Acts = global_weight_preditor.activations

    if args.do_generation == True:
        print('begin generation')
        with torch.no_grad():
            outputs = model(input_ids)
        print('end generation')
    else :
        pass

    print('sp_configs: ',sp_configs)
    for sp_config in sp_configs:
        print('begin threshold', sp_config)
        global_weight_preditor.activations.find_threshold(sp_config[0], os.path.join(args.file_path,'threshold'))
        print('end threshold', sp_config)


if __name__ == "__main__":
    main()