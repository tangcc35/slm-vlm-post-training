#!/usr/bin/env python3
"""Ad-hoc GGUF inference script using llama-cpp-python.

Configure parameters via the UPPERCASE constants below.
"""

import sys
from llama_cpp import Llama

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_PATH = "outputs/qwen35_08b_nsfw_story/gguf/Qwen3.5-0.8B.Q8_0.gguf"
SYSTEM_PROMPT = "You are a creative writer."
PROMPT = "Write a short opening scene for a mystery story."

TEMPERATURE = 0.7
TOP_P = 0.9
REPEAT_PENALTY = 1.1
MAX_TOKENS = 512
N_GPU_LAYERS = -1   # -1 offloads all layers to GPU; 0 runs purely on CPU
N_CTX = 2048
STREAM = True
VERBOSE = False
# =====================================================================


def main() -> None:
    print(f"Loading GGUF model: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=VERBOSE,
    )

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": PROMPT})

    print(f"\nUser Prompt: {PROMPT}")
    print("\n--- Output ---\n")

    if STREAM:
        response_stream = llm.create_chat_completion(
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for chunk in response_stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                print(delta["content"], end="", flush=True)
        print()
    else:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
