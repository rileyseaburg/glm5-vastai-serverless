"""
GLM-5 FP8 PyWorker for Vast.ai Serverless
==========================================
Exposes OpenAI-compatible endpoints for CodeTether agents:
  - /v1/chat/completions  (primary - agentic tool use)
  - /v1/completions       (legacy completions)

Model: zai-org/GLM-5-FP8 (744B MoE, 40B active params)
Backend: vLLM with MTP speculative decoding
GPU: 8x A100 SXM4 80GB

CodeTether Integration:
  Workers point base_url at the Vast serverless endpoint URL.
  It's OpenAI-compatible — no client changes needed.
"""

import os
import random
import string

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

# ---------------------------------------------------------------------------
# Model server config
# The start-server script launches vLLM on this port.
# PyWorker proxies requests to it.
# ---------------------------------------------------------------------------

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", 18000))
MODEL_LOG_FILE    = os.environ.get("MODEL_LOG_FILE", "/var/log/portal/vllm.log")
MODEL_NAME        = os.environ.get("MODEL_NAME", "glm-5-fp8")

# ---------------------------------------------------------------------------
# Workload calculators
# These feed the autoscaler. For agentic coding workloads, token count
# is the right cost proxy — it determines how many concurrent agents
# a single GPU worker can serve.
# ---------------------------------------------------------------------------

def chat_workload(payload: dict) -> float:
    """Estimate workload for chat completions.

    For CodeTether agents: most requests are tool calls with short contexts
    (2-4k input) and moderate output (512-2k tokens). We use max_tokens as
    the primary signal since that's what consumes GPU compute time.

    If messages are provided, we add a rough prompt token estimate so the
    autoscaler knows about long-context requests too.
    """
    max_tokens = float(payload.get("max_tokens", 1024))

    # Rough prompt token estimate from messages
    messages = payload.get("messages", [])
    prompt_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
    prompt_tokens = prompt_chars / 4.0  # ~4 chars per token approximation

    return prompt_tokens + max_tokens


def completions_workload(payload: dict) -> float:
    """Estimate workload for legacy completions endpoint."""
    max_tokens = float(payload.get("max_tokens", 1024))

    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        prompt_chars = sum(len(p) for p in prompt if isinstance(p, str))
    else:
        prompt_chars = len(prompt)

    prompt_tokens = prompt_chars / 4.0
    return prompt_tokens + max_tokens


# ---------------------------------------------------------------------------
# Benchmark generators
# These run after model load to measure throughput. The serverless engine
# uses the results to right-size hot/cold capacity.
#
# We simulate CodeTether-style agentic requests:
#   - System prompt with tool definitions (~500 tokens)
#   - User message with task description (~200 tokens)
#   - Moderate output (500 tokens)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a software engineering agent. You have access to the following tools:
- bash: Execute shell commands
- file_write: Create or overwrite files
- file_read: Read file contents
- search: Search codebase for patterns
When given a task, plan your approach, then execute it step by step using tool calls.
Respond with tool calls in the format the user specifies."""

BENCHMARK_TASKS = [
    "Create a Python function that validates email addresses using regex. Include type hints and docstring.",
    "Write a bash script that finds all TODO comments in a git repository and outputs them with file paths and line numbers.",
    "Implement a rate limiter class in TypeScript using the token bucket algorithm. Include tests.",
    "Create a Dockerfile for a Node.js application that uses multi-stage builds for production optimization.",
    "Write a SQL migration that adds a users table with proper indexes and constraints for a PostgreSQL database.",
    "Implement a retry decorator in Python with exponential backoff and jitter. Support both sync and async functions.",
    "Create a GitHub Actions workflow that runs tests, builds a Docker image, and deploys to a staging environment.",
    "Write a Rust function that parses a TOML configuration file and validates required fields with proper error handling.",
]


def chat_benchmark_generator() -> dict:
    """Generate a benchmark payload that simulates a CodeTether agent request."""
    task = random.choice(BENCHMARK_TASKS)

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        # Disable thinking mode for benchmarks — faster, more predictable
        "chat_template_kwargs": {"enable_thinking": False},
    }


# ---------------------------------------------------------------------------
# Log action config
# Tells PyWorker how to detect vLLM state from its log output.
# ---------------------------------------------------------------------------

LOG_ACTION_CONFIG = LogActionConfig(
    on_load=[
        # vLLM prints this when the OpenAI-compatible server is ready
        "Application startup complete.",
        # Alternative: some vLLM versions use this
        "Uvicorn running on",
    ],
    on_error=[
        # vLLM process crashed
        "INFO exited: vllm",
        # Engine-level errors (OOM, CUDA errors, etc)
        "RuntimeError: Engine",
        "torch.cuda.OutOfMemoryError",
        "CUDA error",
        "Traceback (most recent call last):",
        # Model loading failures
        "ValueError: Cannot load",
        "OSError: Error no file",
    ],
    on_info=[
        # Model download progress
        '"message":"Download',
        # Weight loading progress
        "Loading model weights",
        "Loading safetensors",
    ],
)

# ---------------------------------------------------------------------------
# Worker config
# ---------------------------------------------------------------------------

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,

    handlers=[
        # /v1/chat/completions — primary endpoint for CodeTether agents
        # This is also the benchmark handler
        HandlerConfig(
            route="/v1/chat/completions",

            # vLLM handles parallel requests via continuous batching
            allow_parallel_requests=True,

            # 90s queue timeout — CodeTether agents retry on 429
            max_queue_time=90.0,

            workload_calculator=chat_workload,

            benchmark_config=BenchmarkConfig(
                generator=chat_benchmark_generator,
                # 8 runs at concurrency 8 to simulate multi-agent load
                runs=8,
                concurrency=8,
            ),
        ),

        # /v1/completions — legacy endpoint, no benchmark
        HandlerConfig(
            route="/v1/completions",
            allow_parallel_requests=True,
            max_queue_time=90.0,
            workload_calculator=completions_workload,
        ),
    ],

    log_action_config=LOG_ACTION_CONFIG,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

Worker(worker_config).run()
