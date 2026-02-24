#!/bin/bash
set -e

echo "Upgrading transformers for Qwen3.5 MoE support..."
pip install --upgrade transformers accelerate

# The model name and TP size are passed as env vars
echo "Starting vLLM for ${MODEL_NAME}..."
vllm serve "Qwen/${MODEL_NAME}" \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
