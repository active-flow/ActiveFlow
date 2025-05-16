#!/bin/bash

export OMP_NUM_THREADS=16

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "NUM_GPUS" $NUM_GPUS

datasets=("alpaca" "wikitext" "c4")
max_sample=8
batch_size=1
# TODO: fix temperature
temperature=0.7

# ATTENTION: teacher_data cannot use sparse
export SPARSE=0

output_path=${OUTPUT_DIR}/datasets/${MODEL_NAME}

cd data/generation


for dataset in "${datasets[@]}"; do
    # torchrun --nproc_per_node=${NUM_GPUS} generate.py \
    #     --base_model ${MODEL_PATH} \
    #     --dataset_name $dataset \
    #     --out_path  $output_path \
    #     --batch_size $batch_size \
    #     --max_sample $max_sample \
    #     --threshold_path ${THRESHOLD_PATH}
    python single_generate.py \
        --base_model ${MODEL_PATH} \
        --dataset_name $dataset \
        --out_path  $output_path \
        --batch_size $batch_size \
        --max_sample $max_sample \
        --temperature $temperature
    done

# merge data
json_paths=()
for dataset in "${datasets[@]}"; do
    json_paths+=("${output_path}/${dataset}_T${temperature}_N1024_S42_${max_sample}.json")
done
product=$(( max_sample * ${#datasets[@]} ))


# TODO : fix name
OUTPUT_JSON="${output_path}/mix_wikitext_alpaca_c4_${product}.json"
python mix_data.py --output "$OUTPUT_JSON" --inputs "${json_paths[@]}"

cd ../..

