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

uv run wandb login
uv run hf auth whoami

echo "********** Starting data curation **********"
uv run slm-post-train curate-data --config configs/sft/qwen35_08b_story_writing.yaml
echo "********** Finished data curation **********"

echo "********** Starting training **********"
uv run slm-post-train train --config configs/sft/qwen35_08b_story_writing.yaml
echo "********** Finished training **********"

echo "********** Starting model export **********"
# Training saves into a timestamped subdirectory (YYYYMMDD-HHMMSS); export the newest one
MODEL_DIR=$(ls -d outputs/qwen35_08b_story_writing/[0-9]* | tail -n 1)
uv run slm-post-train export \
    --model-path "$MODEL_DIR" \
    --output-dir outputs/qwen35_08b_story_writing/gguf \
    --format gguf \
    --quant q8_0
echo "********** Finished model export **********"