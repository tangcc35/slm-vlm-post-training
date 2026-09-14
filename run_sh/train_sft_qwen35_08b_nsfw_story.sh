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
uv run slm-post-train curate-data --config configs/sft/qwen35_08b_nsfw_story.yaml
echo "********** Finished data curation **********"

echo "********** Starting training **********"
uv run slm-post-train train --config configs/sft/qwen35_08b_nsfw_story.yaml
echo "********** Finished training **********"

echo "********** Starting model export **********"
uv run slm-post-train export \
    --model-path outputs/qwen35_08b_nsfw_story \
    --output-dir outputs/qwen35_08b_nsfw_story/gguf \
    --format gguf \
    --quant q8_0
echo "********** Finished model export **********"