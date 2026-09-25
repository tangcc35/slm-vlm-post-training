#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

trap 'kill $(jobs -p) 2>/dev/null' EXIT

# 1. Start llama.cpp model server
MODEL_ALIAS="${STORY_RP_MODEL:-qwen3.5-2b}"
MODEL_ALIAS="${MODEL_ALIAS#*/}"

HOST="${LLAMA_HOST:-127.0.0.1}" PORT="${LLAMA_PORT:-8001}" uv run python -m llama_cpp.server \
    --model "${LLAMA_MODEL_PATH:-outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf}" \
    --model_alias "$MODEL_ALIAS" \
    --host "${LLAMA_HOST:-127.0.0.1}" \
    --port "${LLAMA_PORT:-8001}" \
    --n_gpu_layers "${N_GPU_LAYERS:--1}" \
    --n_ctx "${N_CTX:-4096}" &

until curl -s "http://${LLAMA_HOST:-127.0.0.1}:${LLAMA_PORT:-8001}/v1/models" >/dev/null 2>&1; do
    sleep 1
done

# 2. Start Arize Phoenix Observability (if enabled)
[ "${PHOENIX_ENABLED:-true}" = "true" ] && PHOENIX_HOST="${PHOENIX_HOST:-0.0.0.0}" PHOENIX_PORT="${PHOENIX_PORT:-6006}" phoenix serve &

# 3. Start Backend API
uv run python -m uvicorn story_rp_engine.api.app:create_app \
    --factory \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --reload
