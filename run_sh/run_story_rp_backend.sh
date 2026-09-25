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

# Server configuration with defaults
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
STORY_RP_MODEL="${STORY_RP_MODEL:-ollama/llama3.1:8b}"
STORY_RP_API_BASE="${STORY_RP_API_BASE:-http://localhost:11434}"

echo "***************************************************************"
echo "  Starting Dual-Mode Story & Roleplay Engine Backend Server    "
echo "***************************************************************"
echo "Host:     $HOST"
echo "Port:     $PORT"
echo "Model:    $STORY_RP_MODEL"
echo "API Base: $STORY_RP_API_BASE"
echo "Docs:     http://${HOST}:${PORT}/docs"
echo "Health:   http://${HOST}:${PORT}/health"
echo "***************************************************************"

export STORY_RP_MODEL
export STORY_RP_API_BASE

uv run uvicorn story_rp_engine.api.app:create_app \
    --factory \
    --host "$HOST" \
    --port "$PORT" \
    --reload
