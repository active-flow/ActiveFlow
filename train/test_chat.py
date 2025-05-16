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
from accelerate import infer_auto_device_map, dispatch_model
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from utils.sparse_hook import prepare_sparse_hook
from utils.models import get_llm
from quantization.qlinear import quant_and_dequant_model_q4_0


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--sparse', type=float, default=0,help='sparse')
    parser.add_argument('--do_cr',type=int, default=0,help='cross_layer')
    parser.add_argument('--threshold_path',type=str, default=0,help='threshold_path')
    parser.add_argument('--sparse_strategy',type=str, help='sparse strategy')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of test samples.')
    parser.add_argument('--quant',type=int,default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"EOS token: {tokenizer.eos_token} | ID: {tokenizer.eos_token_id}")
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
    print(f"EOS token: {tokenizer.eos_token} | ID: {tokenizer.eos_token_id}")

    prepare_sparse_hook(model)
    model.eval()
    if args.quant:
        quant_and_dequant_model_q4_0(model)
    
    batch_size = args.batch_size

    while(True) :
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat.")
            break
        else :
            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100,
                    do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id 
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model: {response}")


if __name__ == "__main__":
    main()