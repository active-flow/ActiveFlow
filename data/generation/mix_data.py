import os
import json
import random
import argparse

def mix_json_files(output_path, json_paths):
    """
    读取若干 JSON 文件，将其内容混合后写入到一个新的 JSON 文件中。

    :param output_path: 混合后的输出文件路径
    :param json_paths: 待混合的 JSON 文件路径列表
    """
    all_outputs = []

    # 读取每个 JSON 文件
    for json_path in json_paths:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                dataset_for_eval = f.readlines()
            for line in dataset_for_eval:
                json_data = json.loads(line)
                all_outputs.append(json_data)
        else:
            print(f"Warning: {json_path} does not exist. Skipping...")

    # 混合所有读取的数据
    random.shuffle(all_outputs)

    # 将混合后的数据写入到输出文件中
    print('finish', output_path)
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in all_outputs:
            f.write(json.dumps(item) + '\n')

def main():
    # 使用 argparse 接受命令行参数
    parser = argparse.ArgumentParser(description='Mix JSON files into one output file.')
    parser.add_argument('--output', type=str, required=True, help='Path to output mixed JSON file.')
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help='List of input JSON file paths.')
    
    args = parser.parse_args()

    # 调用函数混合文件
    mix_json_files(args.output, args.inputs)

if __name__ == '__main__':
    main()
