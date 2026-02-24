#!/bin/bash

# Pipe all output to a log file in a directory that survives container restarts
mkdir -p /workspace/logs
LOGFILE="/workspace/logs/onstart_$(date +%s).log"
exec > >(tee -a $LOGFILE) 2>&1

echo "=== Starting GLM-5 Setup ==="
echo "Date: $(date)"
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version)"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade transformers vastai > /dev/null
pip show vllm transformers

# Start vLLM and capture its exit code
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
    --enable-prefix-caching

EXIT_CODE=$?
echo "vLLM exited with code $EXIT_CODE"

# If it fails, keep the container alive for 1 hour so we can SSH in and debug
if [ $EXIT_CODE -ne 0 ]; then
    echo "vLLM failed! Keeping container alive for debugging..."
    sleep 3600
fi
