#!/bin/bash
set -e

# Install vastai SDK for PyWorker
pip install vastai

# Get model name from env, default to GLM-5 if not set
MODEL=${MODEL_NAME:-"glm-5-fp8"}

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
        --reasoning-parser qwen3
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
        --reasoning-parser qwen3
else
    echo "Starting GLM-5..."
    vllm serve zai-org/GLM-5-FP8 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --speculative-config.method mtp \
        --speculative-config.num_speculative_tokens 1 \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        --served-model-name glm-5-fp8 \
        --host 127.0.0.1 \
        --port 18000 \
        --trust-remote-code \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --download-dir /workspace/hf_cache
fi
