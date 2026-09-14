# uv run slm-post-train curate-data --config configs/sft/qwen35_08b_nsfw_story.yaml
# uv run slm-post-train train --config configs/sft/qwen35_08b_nsfw_story.yaml
uv run slm-post-train export \
    --model-path outputs/qwen35_08b_nsfw_story \
    --output-dir outputs/qwen35_08b_nsfw_story/gguf \
    --format gguf \
    --quant q8_0