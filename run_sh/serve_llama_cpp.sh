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

# Defaults - using Q8_0 or Q4_K_M from outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive
DEFAULT_MODEL="outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"
MODEL_PATH="${LLAMA_MODEL_PATH:-$DEFAULT_MODEL}"
HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8001}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}" # -1 offloads all layers to GPU
N_CTX="${N_CTX:-4096}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.5-2b}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at: $MODEL_PATH"
    echo "Available GGUF models in outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive/:"
    ls -lh outputs/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive/*.gguf
    exit 1
fi

echo "***************************************************************"
echo "           Starting llama.cpp Local Model Server               "
echo "***************************************************************"
echo "Model:       $MODEL_PATH"
echo "Alias:       $MODEL_ALIAS"
echo "Host:        $HOST"
echo "Port:        $PORT"
echo "GPU Layers:  $N_GPU_LAYERS"
echo "Context:     $N_CTX"
echo "Endpoint:    http://${HOST}:${PORT}/v1"
echo "***************************************************************"

uv run python -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --n_ctx "$N_CTX"
