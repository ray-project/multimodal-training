#!/bin/bash
# Condition 3: multimodal-training with SP (vision) & TP (language)
# 4 GPUs, parallel_size=4, dp_size=1, local batch=2, global batch=2

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=============================================="
echo "Condition 3: multimodal-training - SP (vision) & TP (language)"
echo "GPUs: 4, parallel_size: 4, dp_size: 1, Batch size: 2"
echo "=============================================="

cd /home/ray/default/multimodal-training

python -m python.train_ray \
    --config-path=../configs \
    --config-name=condition3_sp_tp

echo "Condition 3 completed!"
