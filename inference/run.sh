#!/bin/bash

torchrun --nnodes 1 \
  --nproc-per-node 8 \
  --node-rank 0 \
  --master-addr localhost generate.py \
  --ckpt-path /home/ying.cao/codes/DeepSeek-V3.1-Terminus \
  --config configs/config_671B.json \
  --interactive --temperature 0.7 --max-new-tokens 200 2>&1 | tee run.log