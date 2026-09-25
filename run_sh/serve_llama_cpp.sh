#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

trap 'kill $(jobs -p) 2>/dev/null' EXIT

# 1. Start llama.cpp model server
HOST="$LLAMA_HOST" PORT="$LLAMA_PORT" uv run python -m llama_cpp.server \
    --model "$LLAMA_MODEL_PATH" \
    --model_alias "${STORY_RP_MODEL#*/}" \
    --host "$LLAMA_HOST" \
    --port "$LLAMA_PORT" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --n_ctx "$N_CTX" &

until curl -s "http://$LLAMA_HOST:$LLAMA_PORT/v1/models" >/dev/null 2>&1; do
    sleep 1
done

# 2. Start Arize Phoenix Observability (if enabled)
[ "$PHOENIX_ENABLED" = "true" ] && uv run phoenix serve &

# 3. Start Backend API
uv run python -m uvicorn story_rp_engine.api.app:create_app \
    --factory \
    --host "$HOST" \
    --port "$PORT" \
    --reload
