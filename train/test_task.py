import os
import argparse
import numpy as np
import fnmatch
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Function
from transformers import AutoModelForCausalLM,AutoTokenizer
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from utils.sparse_hook import prepare_sparse_hook
from utils.models import get_llm
from quantization.qlinear import quant_and_dequant_model_q4_0
from train.utils import make_supervised_data_module, smart_tokenizer_and_embedding_resize

from peft import PeftModel

def str_to_list(value):
    return value.split(',')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--task", type=str_to_list, help="List of tasks to evaluate")
    parser.add_argument('--limit', type=int, default=-1, help='Number of test samples.')
    parser.add_argument('--num_shot',type=int, default=0,help='NUM_SHOT')
    parser.add_argument('--sparse', type=float, default=0,help='sparse')
    parser.add_argument('--do_cr',type=int, default=0,help='cross_layer')
    parser.add_argument('--threshold_path',type=str, default=0,help='threshold_path')
    parser.add_argument('--sparse_strategy',type=str, help='sparse strategy')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of test samples.')
    parser.add_argument('--test_all',type=int,default=0)
    parser.add_argument('--quant',type=int,default=0)
    parser.add_argument('--use_lora',type=int,default=0)
    parser.add_argument('--lora_checkpoint', type=str, default="", help='LLaMA model.')
    args = parser.parse_args()
    print('-'*40)
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('-'*40)
    return args

def download_dataset(task_name):
    import datasets
    datasets.load_dataset(
        path=task_name,
        split="validation",
        name=None,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        trust_remote_code=True
    )

def eval(
    model,
    task_list=["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"],
    num_shot=0,
    batch_size=1,
    limit=None
):
    #from .llm_eval_adaptor import LMEvalAdaptor
    from lm_eval import evaluator
    from lm_eval.utils import make_table
    from lm_eval.tasks import TaskManager
    from lm_eval.models.huggingface import HFLM
    task_manager = TaskManager()
    task_names = task_manager.match_tasks(task_list)
    task_num_fewshot = {
        "agieval": 0,
        "mmlu": 5,
        "arc_challenge": 25,
    }
    lm_eval_model = HFLM(model, batch_size=batch_size, max_length=2048)
    results = []
    for task_name in task_names:
        num_fewshot = task_num_fewshot.get(task_name, num_shot)

        try:
        # Evaluate the current task
            task_result = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                use_cache=None,
                limit=limit,
                check_integrity=False,
                verbosity='ERROR'
            )
        except Exception as e:
            print(f"Task '{task_name}' testing failed: {e}")

        else :
            print('finish_task', task_name)
            print(make_table(task_result))
            results.append(make_table(task_result))

    # 生成结果表格
    return results

def process_compressed_pred(compressed_pred, output_file, layer_idx, weight_idx):
    """
    test
    """
    sorted_pred, indices = torch.sort(compressed_pred, descending=True)
    
    top_k = int(0.2 * compressed_pred.numel())
    top_elements = sorted_pred[:top_k]
    
    total_sum = compressed_pred.sum().item()
    top_sum = top_elements.sum().item()
    top_ratio = top_sum / total_sum if total_sum > 0 else 0

    with open(output_file, 'a') as f:
        f.write(f"Layer {layer_idx}, Weight {weight_idx}:\n")
        f.write(f"Top 20% elements: {top_elements.tolist()}\n")
        f.write(f"Top 20% ratio: {top_ratio:.4f}\n")
        f.write(f"Total sum: {total_sum:.4f}\n")
        f.write(f"Top sum: {top_sum:.4f}\n")
        f.write("-" * 40 + "\n")
    
    print(f"Layer {layer_idx}, Weight {weight_idx}:")
    # print(f"Top 20% elements: {top_elements.tolist()}")
    print(f"Top 20% ratio: {top_ratio:.4f}")
    print(f"Total sum: {total_sum:.4f}")
    print(f"Top sum: {top_sum:.4f}")
    print("-" * 40)

def debug_test(model):
    global_weight_preditor = model.predictor
    print('DEBUG_CROSSLAYER', os.environ.get('DEBUG_CROSSLAYER','0'))
    if os.environ.get('DEBUG_CROSSLAYER','0') != '0' :
        print(f"Total elements: {global_weight_preditor.sparse_params[0]}")
        print(f"Zero elements: {global_weight_preditor.sparse_params[1]}")
        print(f"Zero ratio: {global_weight_preditor.sparse_params[1] / global_weight_preditor.sparse_params[0]:.4f}")
        with open('test_crosslayer.txt', 'w') as f:
            pass  # 清空文件内容
        predictor = model.predictor
        for layer_idx in range(predictor.num_layers):
            for weight_idx in range(predictor.weight_counters[layer_idx]):
                if predictor.preds[layer_idx][weight_idx] != []:
                    size = len(predictor.preds[layer_idx][weight_idx][0])
                    metric = np.zeros((size, size))
                    print(predictor.preds[layer_idx][weight_idx])
                    for i in range(len(predictor.preds[layer_idx][weight_idx])):
                        if predictor.preds[layer_idx][weight_idx][i] is not None:
                            pred = predictor.preds[layer_idx][weight_idx][i]
                            print(pred)
                            for j in range(size):
                                for k in range(size):
                                    if pred.item(j) == pred.item(k):
                                        metric[j][k] += 1
                    metric = metric / len(predictor.preds[layer_idx][weight_idx])
                    metric = np.triu(metric, k=1)
                    print(f"Layer {layer_idx}, Weight {weight_idx}:")
                    print(f"Metric shape: {metric.shape}")
                    print(f"Metric: {metric}")
                    print("-" * 40)
                    


def eval_for_sp_config(model_path, model, task_list, num_shot, batch_size, limit, sp_config, threshold_path, strategy):
    print("begin : sp_config: ", sp_config, "num_shot: ", num_shot)

    # set sparsity
    global_weight_preditor = model.predictor
    global_weight_preditor.reset()
    attn_sp, mlp_sp, w_p, do_cr = sp_config
    global_weight_preditor.set_sp_config(attn_sp, mlp_sp, w_p)
    global_weight_preditor.set_do_pre_prediction(do_cr)
    global_weight_preditor.set_sparsity_threshold(threshold_path)
    global_weight_preditor.set_sparsity_strategy(strategy)

    # eval
    results = eval(model, task_list, num_shot, batch_size, limit)
    
    print('='*40)
    print("end : sp_config: ", sp_config, "num_shot: ", num_shot)
    print("results")
    for result in results :
        print(result)
    debug_test(model)
    print('='*40)
    


def main():
    args = parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model)
    prepare_sparse_hook(model)

    # TODO: add quant qrgs
    if args.quant:
        quant_and_dequant_model_q4_0(model)
    # for train: load smart eos
    # use lora
    if args.use_lora :
        DEFAULT_PAD_TOKEN = "[PAD]"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            print("tokenizer has not padding token")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        print("Loading LoRA adapter weights ...")
        # wrapper fpr Lora
        model = PeftModel.from_pretrained(model, args.lora_checkpoint)
        print(model)
        for name, module in model.named_modules():
            if "q_proj" not in name:
                module.requires_grad = False
        model = model.merge_and_unload()

    
    model.eval()


    print("task: eval")
    
    print(f"task_name: {args.task}")
    
    limit = None if ( args.limit is None or args.limit < 0 ) else args.limit
    
    batch_size = args.batch_size
    
    task_list = [*args.task]
        
    print('task_list',task_list)
    sp_configs = [(args.sparse, args.sparse, 0.00, args.do_cr)]
    if args.test_all:
        sp_configs = [(0,0,0,0), (0.5,0.5,0,0), (0.6,0.6,0,0), (0.7,0.7,0,0), (0.8,0.8,0,0),(args.sparse, args.sparse, 0.00, args.do_cr)]
        sp_configs = list(set(sp_configs))
        
    num_shots= [args.num_shot]
    
    print('sp_configs: ',sp_configs)
    for sp_config in sp_configs:
        for num_shot in num_shots: 
            if args.test_all:
                data_dir = os.environ.get('DATA_DIR')
                model_name = os.environ.get('MODEL_NAME')
                sparse = sp_config[0]
                threshold_path = f'{data_dir}/threshold/{model_name}/sparse-{sparse}.json'
            else :
                threshold_path = args.threshold_path
            eval_for_sp_config(args.model, model, task_list, num_shot, batch_size, limit, sp_config, threshold_path, args.sparse_strategy)


if __name__ == "__main__":
    main()