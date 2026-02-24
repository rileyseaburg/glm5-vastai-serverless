#!/bin/bash
set -e

# Install vastai SDK for PyWorker
pip install vastai

# Get model name from env, default to GLM-5 if not set
MODEL=${MODEL_NAME:-"glm-5-fp8"}
mkdir -p /workspace/hf_cache

# We must run vllm in the background so the Vast container stays alive
if [[ "$MODEL" == "Qwen3.5-35B-A3B" ]]; then
    echo "Starting Qwen 35B..."
    vllm serve Qwen/Qwen3.5-35B-A3B \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --port 18000 \
        --download-dir /workspace/hf_cache \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 &
elif [[ "$MODEL" == "Qwen3.5-122B-A10B" ]]; then
    echo "Starting Qwen 122B..."
    vllm serve Qwen/Qwen3.5-122B-A10B \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --port 18000 \
        --download-dir /workspace/hf_cache \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 &
fi

# Keep container alive
tail -f /dev/null
