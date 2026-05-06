#!/bin/bash

MASTER_ADDR='localhost'
MASTER_PORT=$(shuf -i 20000-29999 -n 1)

# 模型列表
declare -A MODELS=(
    ["qwen2.5"]="/root/hf-models/Qwen2.5-3B-Instruct"
)

# 数据集列表
declare -A DATASETS=(
    ["nwgi"]="/root/deca/data/nwgi"
)

# Dirichlet α 列表
DIRICHLET_LIST=(0.25)

run_job() {
    local model=$1
    local model_path=$2
    local dataset=$3
    local data_path=$4
    local alpha=$5

    echo ">>> Running: model=$model  dataset=$dataset  alpha=$alpha"

    python -m torch.distributed.run \
            --nproc_per_node=8 \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            main.py \
            --method_type deca \
            --model_type $model \
            --base_model_path $model_path \
            --num_clients 8 \
            --num_rounds 256 \
            --topology Random \
            --optim blockwise \
            --learning_rate 5e-5 \
            --local_epochs 24 \
            --comm_frq 1 \
            --blkcg_frq 60 \
            --dataset_name $dataset \
            --data_path $data_path \
            --eval_strategy no \
            --max_len 512 \
            --div_test_data True \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 64 \
            --lora_rank 64 \
            --dirichlet_alpha $alpha \
            --blk_seq descending
}

# ==================================
# 主循环：模型 × 数据集 × alpha
# ==================================

for model in "${!MODELS[@]}"; do
    for dataset in "${!DATASETS[@]}"; do
        for alpha in "${DIRICHLET_LIST[@]}"; do
            run_job "$model" "${MODELS[$model]}" "$dataset" "${DATASETS[$dataset]}" "$alpha"
        done
    done
done
