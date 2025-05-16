from transformers import AutoModelForCausalLM
import time
from accelerate import infer_auto_device_map, dispatch_model
import os
import torch
from transformers import AutoModelForCausalLM

def get_available_gpu_memory(device=0):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)  # 返回值以字节为单位
        free_gib = free / (1024 ** 3)
        total_gib = total / (1024 ** 3)
        return free_gib, total_gib
    else:
        return None, None

def get_max_memory():
    max_memory = {}
    num_devices = torch.cuda.device_count()
    for device in range(num_devices):
        free_gib, total_gib = get_available_gpu_memory(device)
        if free_gib is not None:
            allocation = free_gib * 0.95  # 分配90%空闲内存
            max_memory[device] = f"{allocation:.0f}GiB"
    max_memory["cpu"] = "128GiB"
    return max_memory

def get_memory_and_device(model) :
    max_memory = get_max_memory()
    print("max_memory:", max_memory)
    
    device_map = infer_auto_device_map(model, max_memory=max_memory)
    print("Device map:", device_map)
    
    return max_memory, device_map

def get_llm(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )

    model.seqlen = model.config.max_position_embeddings 

    model.config.pretraining_tp = 1
    model.config.use_cache = False
    model.config.output_hidden_states = False
    model.config.output_attentions = False

    model.config.do_sample = False
    model.config.temperature = 1.0
    model.config.top_p = 1.0
    model.config._attn_implementation_internal = "eager"

    
    print("model.hf_device_map:", model.hf_device_map)
    return model

def get_auto_batch_size(model, max_batch_size, min_batch_size=1):
    """
    动态调整 batch_size 以适应显存大小。
    
    Args:
        model: 需要进行推理的模型。
        max_batch_size: 最大 batch_size。
        min_batch_size: 最小 batch_size。
    
    Returns:
        合适的 batch_size。
    """
    batch_size = max_batch_size
    while batch_size >= min_batch_size:
        try:
            # f
            dummy_input = torch.ones((batch_size, model.config.max_position_embeddings), dtype=torch.long).to(model.device)
            with torch.no_grad():
                model(dummy_input)
            return batch_size
        except RuntimeError as e:
            # print('error:auto batch', batch_size, e)
            if 'out of memory' in str(e):
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise e
    return min_batch_size