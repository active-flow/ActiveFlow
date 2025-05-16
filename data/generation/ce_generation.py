#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from datasets import load_dataset

def split_text(text, split_ratio=0.2):
    """
    将给定文本按比例分成两部分。

    参数:
        text (str): 原始文本。
        split_ratio (float): 前一部分所占比例（0~1之间），默认为 0.2，即前 20% 作为前缀。

    返回:
        list: 包含两部分的列表 [prefix, continuation]。
    """
    if not text:
        return ["", ""]
    split_index = int(len(text) * split_ratio)
    prefix = text[:split_index].strip()
    continuation = text[split_index:].strip()
    return [prefix, continuation]

def main():
    print("Loading C4 dataset...")
    # 启用流式模式加载 C4 数据集的训练部分
    dataset = load_dataset("c4", "en", split="train", streaming=True, trust_remote_code=True)
    # 直接取前 50000 个样本（根据需要调整数量）
    sampled_dataset = dataset.take(50000)
    sample_list = list(sampled_dataset)
    print(f"Loaded {len(sample_list)} samples.")
    
    processed_samples = []
    print("Processing samples...")
    for sample in sample_list:
        text = sample.get("text", "")
        if len(text) < 100:
            continue
        pair = split_text(text, split_ratio=0.2)
        # 按照示例格式保存为一个 pair，此处保存为 [prefix, continuation]
        processed_samples.append([pair])
    
    # 输出结果为 jsonlines 格式，每一行保存一个样本
    output_file = "c4_processed.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Processed {len(processed_samples)} samples. Saved to {output_file}")

if __name__ == "__main__":
    main()
