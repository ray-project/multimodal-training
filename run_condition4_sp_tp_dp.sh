#!/bin/bash
# Condition 4: multimodal-training with SP+DP (vision) & TP+DP (language)
# 8 GPUs, parallel_size=4, dp_size=2, local batch=1, global batch=2

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "=============================================="
echo "Condition 4: multimodal-training - SP+DP (vision) & TP+DP (language)"
echo "GPUs: 8, parallel_size: 4, dp_size: 2, Batch size: 1"
echo "=============================================="

cd /home/ray/default/multimodal-training

python -m python.train_ray \
    --config-path=../configs \
    --config-name=condition4_sp_tp_dp

echo "Condition 4 completed!"
