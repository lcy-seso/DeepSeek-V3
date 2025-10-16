#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

echo "Starting DeepSeek-V3 func_test with 8 GPUs..."

torchrun --nnodes 1 \
  --nproc-per-node 8 \
  --node-rank 0 \
  func_test.py \
  2>&1 | tee run_func_test.log
