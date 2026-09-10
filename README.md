# SLM & VLM Post-Training

Modular, production-ready framework for post-training Small Language Models (SLMs) and Vision-Language Models (VLMs) using **Unsloth**, **Hugging Face TRL**, **PyTorch**, and **uv**.

Designed from the ground up to train on consumer GPUs with **≤ 8GB VRAM** via 4-bit QLoRA, memory-efficient Unsloth kernels, and paged 8-bit optimizers.

---

## Key Features

- **Responses-Only SFT Loss Masking:** Automatically masks user prompt tokens so gradient backpropagation is computed strictly on model completions using `unsloth.chat_templates.train_on_responses_only`.
- **GRPO Reinforcement Learning:** Full Group Relative Policy Optimization (GRPO) implementation with multi-generation sampling, reference policy KL estimation, and custom scoring functions.
- **Extensible Reward Function Registry:** Modular `@register_reward` decorator system. Ships with standard rewards:
  - `xml_format`: Verifies reasoning structure `<think>...</think><answer>...</answer>`.
  - `exact_match`: Extracts `<answer>...</answer>` content and evaluates against ground-truth answers.
  - `code_execution`: Safely executes Python code blocks inside isolated subprocess workers with timeout enforcement.
- **Comprehensive Model Export & Quantization:**
  - Standard LoRA adapters.
  - 16-bit full merged checkpoints (`merged_16bit`).
  - 4-bit merged checkpoints (`merged_4bit`).
  - GGUF binary quantization (`q4_k_m`, `q8_0`, `f16`, etc.) for local deployment in **Ollama** and **llama.cpp**.
- **Ultra-Low Memory Footprint (≤ 8 GB VRAM):** Native 4-bit quantization (`bitsandbytes`), Unsloth gradient checkpointing, and `adamw_8bit` / `paged_adamw_8bit` optimizers enable training 2B-9B SLMs on single consumer GPUs.
- **Modern Python Tooling:** Fast, reproducible environments managed via `uv` and Python 3.11.

---

## Repository Architecture

```text
slm-vlm-post-training/
├── configs/
│   ├── sft/
│   │   ├── smoke_test.yaml          # Fast 2-step SFT sanity check
│   │   └── gemma_text_sft.yaml      # Gemma 2B SFT recipe (FineTome-100k)
│   └── grpo/
│       ├── smoke_test.yaml          # Fast 2-step GRPO sanity check
│       └── gemma_sudoku_rl.yaml     # Gemma 2B Sudoku reasoning GRPO recipe
├── data/
│   └── dummy/
│       ├── sft_sample.jsonl         # Dummy dataset for SFT testing
│       └── grpo_sample.jsonl        # Dummy dataset for GRPO testing
├── src/
│   └── slm_post_train/
│       ├── cli.py                   # Unified CLI entrypoint (train, export)
│       ├── data/
│       │   ├── sft_data.py          # Conversational SFT dataset loader & formatter
│       │   └── grpo_data.py         # Prompt-based GRPO dataset loader & normalizer
│       ├── export/
│       │   └── exporter.py          # LoRA, merged, and GGUF quantization exporter
│       ├── models/
│       │   └── loader.py            # FastLanguageModel / FastVisionModel loader
│       ├── rewards/
│       │   ├── registry.py          # Extensible reward decorator & registry
│       │   └── standard.py          # XML format, exact match, code execution rewards
│       └── trainers/
│           ├── sft_runner.py        # Unsloth SFT training runner
│           └── grpo_runner.py       # Unsloth GRPO training runner
├── tests/                           # 85+ unit and integration tests
├── pyproject.toml                   # Project dependencies and entrypoints
└── README.md
```

---

## Installation & Quickstart

### Prerequisites

- Linux or WSL2 (Ubuntu 22.04+ recommended)
- NVIDIA GPU with CUDA 12.x support (optional for CPU testing/mocking)
- Python 3.11 (`.python-version` is pinned to 3.11)
- [`uv`](https://docs.astral.sh/uv/) package manager

### Setup Environment

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone repository and sync dependencies:
   ```bash
   git clone https://github.com/your-org/slm-vlm-post-training.git
   cd slm-vlm-post-training
   uv sync --extra dev
   ```

3. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

---

## Running Tests

Run the complete test suite using `pytest`:

```bash
uv run pytest -v
```

Or with active virtual environment:

```bash
pytest -v
```

The test suite covers:
- CLI argument parsing, subcommands, and YAML configuration loading.
- Dataset preparation (JSONL, CSV, Hugging Face Hub, conversational formatting, reasoning column normalization).
- Model and LoRA configuration loaders (text and vision modalities, dtype mappings).
- Reward registry, dispatch, and standard scoring functions (XML regex, exact match, process-sandboxed code execution with timeout protection).
- SFT and GRPO training runners (parameter forwarding, response masking, VRAM metrics tracking).
- Export pipeline (LoRA weights, 16-bit merged, 4-bit merged, GGUF quantization).

---

## Smoke Testing Workflows

Verify your local installation and GPU pipeline with minimal 2-step smoke runs using bundled dummy datasets:

### 1. SFT Smoke Test
```bash
uv run slm-post-train train --config configs/sft/smoke_test.yaml
```
- **Model:** `unsloth/gemma-2-2b-it` (4-bit QLoRA)
- **Dataset:** `data/dummy/sft_sample.jsonl`
- **Steps:** 2 steps
- **Output:** `outputs/sft_smoke_test/`

### 2. GRPO RL Smoke Test
```bash
uv run slm-post-train train --config configs/grpo/smoke_test.yaml
```
- **Model:** `unsloth/gemma-2-2b-it` (4-bit QLoRA)
- **Dataset:** `data/dummy/grpo_sample.jsonl`
- **Rewards:** `xml_format`, `exact_match`
- **Steps:** 2 steps
- **Output:** `outputs/grpo_smoke_test/`

---

## Production Training Workflows

### Supervised Fine-Tuning (SFT)

Run instruction fine-tuning on text models with response token masking:

```bash
uv run slm-post-train train --config configs/sft/gemma_text_sft.yaml
```

**Key SFT Configuration options (`configs/sft/gemma_text_sft.yaml`):**
```yaml
stage: sft

model:
  name_or_path: "unsloth/gemma-2-2b-it"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  use_gradient_checkpointing: "unsloth"

dataset:
  path: "mlabonne/FineTome-100k"
  chat_template: "chatml"
  split: "train[:3000]"

training:
  batch_size: 2
  gradient_accumulation_steps: 4
  max_steps: 100
  learning_rate: 2.0e-4
  optim: "adamw_8bit"
  train_on_responses_only: true     # Masks user prompt tokens during loss calculation

output:
  output_dir: "outputs/gemma_sft"
  logging_steps: 5
```

### Reinforcement Learning (GRPO)

Train reasoning capabilities with Group Relative Policy Optimization and rule-based reward functions:

```bash
uv run slm-post-train train --config configs/grpo/gemma_sudoku_rl.yaml
```

**Key GRPO Configuration options (`configs/grpo/gemma_sudoku_rl.yaml`):**
```yaml
stage: grpo

model:
  name_or_path: "unsloth/gemma-2-2b-it"
  max_seq_length: 1024
  load_in_4bit: true

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

dataset:
  path: "data/dummy/grpo_sample.jsonl"
  split: "train"

rewards:
  - "xml_format"
  - "exact_match"
  - "code_execution"

training:
  batch_size: 1
  gradient_accumulation_steps: 2
  num_generations: 4                # Number of candidate completions per prompt
  max_steps: 50
  max_prompt_length: 256
  max_completion_length: 512
  learning_rate: 5.0e-6

output:
  output_dir: "outputs/gemma_sudoku_grpo"
  logging_steps: 2
```

---

## Exporting Models

The `export` subcommand exports trained checkpoints into standalone artifacts:

```bash
uv run slm-post-train export \
    --model-path <path_or_hf_id> \
    --output-dir <destination_directory> \
    --format <export_format> \
    [--quant <quantization_method>]
```

### Export Formats

| Format | Command Flag | Description | Typical Use Case |
|---|---|---|---|
| **LoRA Adapter** | `--format lora` | Saves lightweight LoRA adapter weights and tokenizer configuration. | Checkpointing, sharing adapter diffs. |
| **Merged 16-bit** | `--format merged_16bit` | Merges LoRA weights back into base model in 16-bit precision (`fp16`/`bf16`). | Full Hugging Face Hub release, vLLM serving. |
| **Merged 4-bit** | `--format merged_4bit` | Merges LoRA weights into a 4-bit quantized base model. | Low-memory Hugging Face Transformers inference. |
| **GGUF** | `--format gguf` | Exports model directly into a quantized `.gguf` binary. | **Ollama**, **llama.cpp**, LM Studio. |

### GGUF Quantization Examples

Export to `q4_k_m` (default recommended balance between size and quality):
```bash
uv run slm-post-train export \
    --model-path outputs/gemma_sft \
    --output-dir exports/gemma_sft_gguf \
    --format gguf \
    --quant q4_k_m
```

Export high-precision `q8_0` or `f16`:
```bash
uv run slm-post-train export \
    --model-path outputs/gemma_sudoku_grpo \
    --output-dir exports/gemma_grpo_q8 \
    --format gguf \
    --quant q8_0
```

### Serving with Ollama

1. Create a `Modelfile`:
   ```dockerfile
   FROM ./exports/gemma_sft_gguf/unsloth.Q4_K_M.gguf

   TEMPLATE """{{ if .System }}<start_of_turn>system
   {{ .System }}<end_of_turn>
   {{ end }}{{ if .Prompt }}<start_of_turn>user
   {{ .Prompt }}<end_of_turn>
   {{ end }}<start_of_turn>model
   {{ .Response }}<end_of_turn>"""

   PARAMETER stop "<start_of_turn>"
   PARAMETER stop "<end_of_turn>"
   PARAMETER temperature 0.7
   ```

2. Register and start model in Ollama:
   ```bash
   ollama create gemma-custom -f Modelfile
   ollama run gemma-custom "Explain how backpropagation works."
   ```

### Serving with llama.cpp

Run inference directly with `llama-cli`:
```bash
llama-cli -m ./exports/gemma_sft_gguf/unsloth.Q4_K_M.gguf \
    -p "<start_of_turn>user\nSolve this puzzle:\n...\n<end_of_turn>\n<start_of_turn>model\n" \
    -n 256
```

---

## Adding Custom Reward Functions

GRPO reward functions compute numerical scalar rewards for model completions. Use the `@register_reward` decorator to expose custom rewards to GRPO configs.

### 1. Define and Register Reward Function

Create a python file or add to `src/slm_post_train/rewards/standard.py`:

```python
from typing import Any, List
from slm_post_train.rewards.registry import register_reward

@register_reward("solution_length_penalty")
def solution_length_penalty_reward(
    prompts: List[Any],
    completions: List[Any],
    max_len: int = 500,
    **kwargs
) -> List[float]:
    """Penalizes overly verbose reasoning chains."""
    scores = []
    for completion in completions:
        # Completions may be raw strings or conversational dict lists
        text = completion if isinstance(completion, str) else str(completion)
        if len(text) <= max_len:
            scores.append(1.0)
        else:
            # Linear decay penalty
            penalty = max(0.0, 1.0 - (len(text) - max_len) / 1000.0)
            scores.append(penalty)
    return scores
```

### 2. Configure in GRPO Recipe

Reference your reward by its registered name in your YAML configuration:

```yaml
stage: grpo

rewards:
  - "xml_format"
  - "exact_match"
  - "solution_length_penalty"
```

The GRPO runner automatically resolves names against the registry at runtime.

---

## 8GB VRAM Optimization Guidelines

When training on 8GB VRAM consumer GPUs (e.g., RTX 3070, RTX 4060, laptop GPUs):

| Parameter | Recommended Setting | Rationale |
|---|---|---|
| `model.load_in_4bit` | `true` | Reduces 2B model footprint from ~5GB to ~1.8GB VRAM. |
| `lora.use_gradient_checkpointing` | `"unsloth"` | Unsloth's optimized checkpointing saves >60% activation memory. |
| `training.batch_size` | `1` | Minimizes peak batch activation overhead. |
| `training.gradient_accumulation_steps` | `2` to `8` | Simulates effective batch size of 2–8 without VRAM penalty. |
| `training.optim` | `adamw_8bit` or `paged_adamw_8bit` | Offloads optimizer states to system RAM if VRAM pressure peaks. |
| `training.max_prompt_length` | `256` (GRPO) | Keeps prompt KV cache minimal during multi-generation rollouts. |
| `training.max_completion_length` | `256` - `512` | Restricts reasoning token sampling buffer. |
| `training.num_generations` | `2` to `4` (GRPO) | Balances policy group advantage estimation with memory usage. |

---

## Development & Contributing

### Formatting & Linting
Ensure code style consistency before submitting pull requests:
```bash
uv run pytest tests/ -v
```

### License
Apache 2.0. See [LICENSE](LICENSE) for details.
