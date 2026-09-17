#!/usr/bin/env bash
set -e

# Determine repository root and load .env if present
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
    echo "Loading environment variables from $ROOT_DIR/.env"
    set -a
    # shellcheck disable=SC1091
    . "$ROOT_DIR/.env"
    set +a
fi

cd "$ROOT_DIR"

# Model Server Configuration
DEFAULT_MODEL="outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"
MODEL_PATH="${LLAMA_MODEL_PATH:-$DEFAULT_MODEL}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8001}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}" # -1 offloads all layers to GPU
N_CTX="${N_CTX:-4096}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.5-2b}"

# Backend API Configuration
APP_HOST="${HOST:-0.0.0.0}"
APP_PORT="${PORT:-8000}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at: $MODEL_PATH"
    echo "Available GGUF models in outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive/:"
    ls -lh outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive/*.gguf
    exit 1
fi

echo "==============================================================="
echo "  [1/2] Starting llama.cpp Local Model Server in Background   "
echo "==============================================================="
echo "Model:       $MODEL_PATH"
echo "Alias:       $MODEL_ALIAS"
echo "GPU Layers:  $N_GPU_LAYERS"
echo "Context:     $N_CTX"
echo "Endpoint:    http://${LLAMA_HOST}:${LLAMA_PORT}/v1"
echo "==============================================================="

# Launch llama_cpp.server in the background
uv run python -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --host "$LLAMA_HOST" \
    --port "$LLAMA_PORT" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --n_ctx "$N_CTX" &
LLAMA_PID=$!

# Trap signals to ensure the background model server is terminated on exit
cleanup() {
    echo ""
    echo "Shutting down model server (PID $LLAMA_PID)..."
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# Wait for the model server to initialize and respond
echo "Waiting for llama.cpp server to be ready on port $LLAMA_PORT..."
MAX_RETRIES=30
for ((i=1; i<=MAX_RETRIES; i++)); do
    if curl -s "http://${LLAMA_HOST}:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
        echo ">>> llama.cpp model server is ready!"
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "Error: llama_cpp server exited unexpectedly."
        exit 1
    fi
    sleep 1
done

echo ""
echo "==============================================================="
echo "  [2/2] Starting Dual-Mode Story & Roleplay Backend API        "
echo "==============================================================="
echo "Backend:     http://${APP_HOST}:${APP_PORT}"
echo "API Docs:    http://${APP_HOST}:${APP_PORT}/docs"
echo "Health:      http://${APP_HOST}:${APP_PORT}/health"
echo "Connected:   http://${LLAMA_HOST}:${LLAMA_PORT}/v1 (Model: openai/$MODEL_ALIAS)"
echo "==============================================================="

export STORY_RP_MODEL="openai/$MODEL_ALIAS"
export STORY_RP_API_BASE="http://${LLAMA_HOST}:${LLAMA_PORT}/v1"

# Run FastAPI backend in the foreground
uv run uvicorn story_rp_engine.api.app:create_app \
    --factory \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --reload
