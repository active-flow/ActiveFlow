import json

def save_first_8k_lines(input_file, output_file, num_lines=8000):
    """
    只保留输入JSON文件的前8k行，并将结果保存到另一个文件中。

    参数:
    - input_file: 输入的JSON文件路径
    - output_file: 输出的JSON文件路径
    - num_lines: 要保留的行数 (默认为8000行)
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        # 逐行读取JSON文件
        lines = []
        for i, line in enumerate(infile):
            if i >= num_lines:
                break
            lines.append(line)
    
    # 写回前8k行数据到新文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(lines)

# 示例调用
input_path = '/home/donglinbai/Projects/xxx/BitDistiller-Q4_0/data/datasets/Llama-2-13b-chat-hf/mix_wikitext_alpaca_c4_9000.json'  # 输入文件路径
output_path = '/home/donglinbai/Projects/xxx/BitDistiller-Q4_0/data/datasets/Llama-2-13b-chat-hf/mix_wikitext_alpaca_c4_8000.json'  # 输出文件路径
save_first_8k_lines(input_path, output_path)