#!/bin/bash
# Condition 3: multimodal-training with DeepSpeed (vision) & DeepSpeed (language)
# 4 GPUs, parallel_size=4, dp_size=1, local batch=2, global batch=2

CONFIG=$1

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=============================================="
echo "Condition: multimodal-training - $CONFIG"
echo "GPUs: 4, parallel_size: 1, dp_size: 4, Batch size: 1"
echo "=============================================="

cd /home/ray/default/multimodal-training

python -m python.train_ray \
    --config-path=../configs \
    --config-name=$CONFIG

echo "Condition $CONFIG completed!"