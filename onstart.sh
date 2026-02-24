#!/bin/bash
set -e

# Install vastai SDK for PyWorker
pip install vastai

# Launch vLLM server exposed to 0.0.0.0 so we can hit it directly via the mapped port
vllm serve zai-org/GLM-5-FP8 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-5-fp8 \
    --host 0.0.0.0 \
    --port 18000 \
    --trust-remote-code \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --download-dir /workspace/hf_cache
