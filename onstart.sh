#!/bin/bash
set -e

# Log everything to a file we can inspect
exec > >(tee -a /var/log/onstart.log) 2>&1

echo "Starting setup at $(date)"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade transformers vastai

# Start vLLM in the background
echo "Starting vLLM..."
vllm serve zai-org/GLM-5-FP8 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-5-fp8 \
    --host 0.0.0.0 \
    --port 18000 \
    --trust-remote-code \
    --max-model-len 32768 \
    --enable-prefix-caching &

VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID"

# Keep script running so container doesn't exit
wait $VLLM_PID
