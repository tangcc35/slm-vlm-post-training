#!/usr/bin/env bash
set -e

trap 'kill $(jobs -p) 2>/dev/null' EXIT

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

# Only default API base for Ollama or local endpoints when not a Gemini model
case "$STORY_RP_MODEL" in
    gemini*|google/*)
        STORY_RP_API_BASE="${STORY_RP_API_BASE:-}"
        ;;
    *)
        STORY_RP_API_BASE="${STORY_RP_API_BASE:-http://localhost:11434}"
        ;;
esac

echo "***************************************************************"
echo "  Starting Dual-Mode Story & Roleplay Engine Backend Server    "
echo "***************************************************************"
echo "Host:     $HOST"
echo "Port:     $PORT"
echo "Model:    $STORY_RP_MODEL"
if [ -n "$STORY_RP_API_BASE" ]; then
    echo "API Base: $STORY_RP_API_BASE"
else
    echo "API Base: Direct Provider Endpoint (Google AI Studio / Cloud)"
fi
echo "UI:       http://${HOST}:${PORT}/"
echo "Docs:     http://${HOST}:${PORT}/docs"
echo "Health:   http://${HOST}:${PORT}/health"
if [ "$PHOENIX_ENABLED" = "true" ] || [ "$PHOENIX_ENABLED" = "1" ]; then
    echo "Phoenix:  http://${PHOENIX_HOST:-127.0.0.1}:${PHOENIX_PORT:-6006}"
fi
echo "***************************************************************"

export STORY_RP_MODEL
if [ -n "$STORY_RP_API_BASE" ]; then
    export STORY_RP_API_BASE
else
    unset STORY_RP_API_BASE
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    export GOOGLE_API_KEY
elif [ -n "$GEMINI_API_KEY" ]; then
    export GOOGLE_API_KEY="$GEMINI_API_KEY"
    export GEMINI_API_KEY
fi

# Start Arize Phoenix Observability in background (if enabled and not already running)
if [ "$PHOENIX_ENABLED" = "true" ] || [ "$PHOENIX_ENABLED" = "1" ]; then
    PHOENIX_CHECK_PORT="${PHOENIX_PORT:-6006}"
    if ! curl -s "http://127.0.0.1:${PHOENIX_CHECK_PORT}" >/dev/null 2>&1; then
        uv run phoenix serve &
    fi
fi

uv run python -m uvicorn story_rp_engine.api.app:create_app \
    --factory \
    --host "$HOST" \
    --port "$PORT" \
    --reload


