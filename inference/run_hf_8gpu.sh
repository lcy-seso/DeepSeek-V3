#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

MODEL_PATH="/data2/shared/dpsk_weights/DeepSeek-V3.1-Terminus"

MAX_NEW_TOKENS=200
TEMPERATURE=0.7
LAYER_INDEX=0  # Load only layer 0 (first layer) - set to null to load full model

echo "Starting DeepSeek-V3 with 8 GPUs..."
echo "Model path: $MODEL_PATH"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Layer index: $LAYER_INDEX"

# Build command with optional layer index
CMD_ARGS="--model-path \"$MODEL_PATH\" \
  --device-map \"auto\" \
  --max-new-tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --interactive"

# Add layer index if specified
if [ ! -z "$LAYER_INDEX" ]; then
  CMD_ARGS="$CMD_ARGS --layer-index $LAYER_INDEX"
fi

torchrun --nnodes 1 \
  --nproc-per-node 8 \
  --node-rank 0 \
  --master-addr localhost \
  --master-port 29500 \
  generate_hf.py \
  $CMD_ARGS \
  2>&1 | tee run_hf.log
