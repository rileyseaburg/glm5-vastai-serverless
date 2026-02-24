# GLM-5 FP8 — Vast.ai Serverless PyWorker

Deploys GLM-5 (744B MoE, 40B active) as a Vast.ai Serverless endpoint with
OpenAI-compatible API for CodeTether agent orchestration.

## What This Does

- Exposes `/v1/chat/completions` and `/v1/completions` via Vast Serverless
- Auto-scales GPU workers based on agent request load
- Benchmarks on startup to calibrate autoscaler for agentic coding workloads
- Scales to zero when idle (you only pay storage, not compute)
- Handles 4-8 concurrent CodeTether agents per GPU worker via continuous batching

## Vast.ai Serverless Setup

### 1. Create Serverless Endpoint

In the Vast.ai console:

- **Template**: Use the vLLM template (or custom template with vLLM)
- **GPU Type**: A100 SXM4 80GB
- **GPU Count**: 8 per worker
- **Min Workers**: 0 (scale to zero when idle)
- **Max Workers**: Set based on budget

### 2. Environment Variables

Set these in your endpoint configuration:

```
MODEL_NAME=glm-5-fp8
MODEL_SERVER_PORT=18000
MODEL_LOG_FILE=/var/log/portal/vllm.log
PYWORKER_REPO=https://github.com/<your-org>/glm5-vastai-serverless
HF_TOKEN=<your-huggingface-token>
```

### 3. vLLM Start Command

Your template's start-server script should launch vLLM with:

```bash
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
    --enable-prefix-caching
```

### 4. Push This Repo

```bash
git init
git add worker.py requirements.txt README.md
git commit -m "GLM-5 Vast.ai Serverless PyWorker"
git remote add origin https://github.com/<your-org>/glm5-vastai-serverless.git
git push -u origin main
```

## CodeTether Integration

Point your CodeTether worker config at the Vast.ai endpoint:

```
model_ref: "glm5:glm-5-fp8"
base_url: "https://<your-vast-endpoint-url>/v1"
```

The endpoint is fully OpenAI-compatible. No client changes needed.

### Disabling Thinking Mode

For faster agent responses (no chain-of-thought overhead), add to requests:

```json
{
    "chat_template_kwargs": {"enable_thinking": false}
}
```

## Cost Model

| State | GPU Compute | Storage | Notes |
|-------|------------|---------|-------|
| Active (serving requests) | ~$7.48/hr | Billed | 4-8 concurrent agents |
| Stopped (no requests) | $0 | Billed (pennies) | Auto-scales down |
| Destroyed | $0 | $0 | Full stop |

Effective cost per agent-hour at 4 concurrent: ~$1.87/hr
Effective cost per agent-hour at 8 concurrent: ~$0.94/hr

Compare: GLM-5 API at heavy usage easily exceeds $10/hr for 4 agents.

## Files

- `worker.py` — PyWorker config (routes, workload calc, benchmarks, log detection)
- `requirements.txt` — Python deps (minimal, runtime provides vastai SDK)
- `README.md` — This file
