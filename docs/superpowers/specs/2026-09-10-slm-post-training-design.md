# Design Specification: SLM Post-Training with Unsloth & uv

- **Date**: 2026-09-10
- **Status**: Approved
- **Target Repository**: `slm-vlm-post-training`

## 1. Overview & Goals

This project establishes a clean, modular repository for post-training Small Language Models (SLMs) using [Unsloth](https://github.com/unslothai/unsloth) and managed via [`uv`](https://github.com/astral-sh/uv).

The design transitions reference code from experimental Colab notebooks (`Gemma4_(E2B)_Text.ipynb` and `Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb`) into an extensible, production-ready codebase supporting:
1. **Supervised Fine-Tuning (SFT)** with chat template standardization and response-only loss masking.
2. **Reinforcement Learning via GRPO (Group Relative Policy Optimization)** with modular reward functions (XML formatting, exact match, code execution).
3. **Model Merging & Quantization Export** (16-bit, 4-bit, and GGUF quantization).
4. **Smoke-testable local workflows** via dedicated dummy datasets and quick test configurations.
5. **Future-ready architecture** for Vision-Language Models (VLMs).

## 2. Environment & Packaging (`uv`)

- **Python Version**: Pinned to Python 3.11 via `.python-version` for broad compatibility with PyTorch, CUDA 12.x wheels, Unsloth, and Triton/FlashAttention.
- **Build System**: `hatchling` via standard `pyproject.toml`.
- **Dependencies**:
  - Deep Learning Core: `torch`, `unsloth`, `trl`, `transformers`, `datasets`, `accelerate`, `bitsandbytes`.
  - Configuration & Utilities: `pyyaml`.
  - Development / Testing: `pytest`.
- **CLI Registration**:
  - `slm-post-train = "slm_post_train.cli:main"`

## 3. Repository Directory Structure

```text
slm-vlm-post-training/
├── pyproject.toml              # uv-managed package metadata, dependencies & CLI entrypoints
├── .python-version             # Pinned Python version (3.11)
├── .gitignore                  # Ignores .venv, outputs, checkpoints, logs, datasets, :Zone.Identifier
├── README.md                   # Setup guide, run commands, configuration documentation
├── data/
│   └── dummy/
│       ├── sft_sample.jsonl    # Lightweight multi-turn conversation dataset for SFT smoke test
│       └── grpo_sample.jsonl   # Lightweight prompt-solution dataset for GRPO smoke test
├── configs/                    # User-facing YAML configurations
│   ├── sft/
│   │   ├── gemma_text_sft.yaml # Production SFT recipe for Gemma SLMs
│   │   └── smoke_test.yaml     # 2-step SFT smoke test config
│   └── grpo/
│       ├── gemma_sudoku_rl.yaml# GRPO RL recipe for Sudoku / reasoning
│       └── smoke_test.yaml     # 2-step GRPO smoke test config
├── src/
│   └── slm_post_train/
│       ├── __init__.py
│       ├── cli.py              # Main CLI entrypoint (train / export commands)
│       ├── models/
│       │   ├── __init__.py
│       │   └── loader.py       # FastLanguageModel & FastVisionModel initialization, LoRA setup
│       ├── data/
│       │   ├── __init__.py
│       │   ├── sft_data.py     # Chat template application, formatting, and dataset loading
│       │   └── grpo_data.py    # Prompt extraction and formatting for RL rollouts
│       ├── rewards/
│       │   ├── __init__.py
│       │   ├── registry.py     # Decorator-based reward function registry
│       │   └── standard.py     # XML format reward, exact match reward, code execution reward
│       ├── trainers/
│       │   ├── __init__.py
│       │   ├── sft_runner.py   # SFT pipeline runner (SFTTrainer + response-only loss)
│       │   └── grpo_runner.py  # GRPO pipeline runner (GRPOTrainer + policy rollouts)
│       └── export/
│           ├── __init__.py
│           └── exporter.py     # Adapter save, 16-bit / 4-bit merge, GGUF quantization
├── tests/
│   ├── test_rewards.py         # Reward function unit tests
│   └── test_data.py            # Data formatting and chat template unit tests
└── unsloth_notebook_reference/ # (Preserved) Reference Colab notebooks
```

## 4. Component Details

### 4.1. Configuration Loading (Inlined in `cli.py`)
- Direct `yaml.safe_load(open(config_path))` directly inside the CLI entrypoint.
- No separate config module or extra layer of indirection needed. Passed directly as plain dictionaries into runners.

### 4.2. Model & LoRA Loader (`src/slm_post_train/models/loader.py`)
- Wraps `unsloth.FastLanguageModel.from_pretrained` (default) with branching hooks for `unsloth.FastVisionModel` (`modality: "vision"`).
- Automatically applies PEFT LoRA with parameters:
  - `r`: Rank (e.g. 16 or 32).
  - `lora_alpha`: Rank $\times 2$ for training acceleration.
  - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
  - `load_in_4bit`: True by default for 8GB VRAM compatibility.
  - `use_gradient_checkpointing`: `"unsloth"` for minimal VRAM footprint.

### 4.3. Data Preparation (`src/slm_post_train/data/`)
- **SFT (`sft_data.py`)**:
  - Ingests local files (`.jsonl`, `.json`, `.csv`) or HuggingFace hub IDs.
  - Applies chat template via `get_chat_template` and standardizes datasets with `standardize_data_formats`.
  - Integrates `train_on_responses_only` to train exclusively on assistant generation tokens.
- **GRPO (`grpo_data.py`)**:
  - Ingests problem/prompt datasets.
  - Formats conversational prompts ready for multi-completion rollout in `GRPOTrainer`.

### 4.4. Reward Registry for GRPO (`src/slm_post_train/rewards/`)
- Registry decorator `@register_reward(name)` manages available reward scorers.
- Standard built-in reward functions:
  - `xml_format_reward`: Validates presence of reasoning and answer tags (e.g. `<think>`, `</think>`, `<answer>`).
  - `exact_match_reward`: Normalizes response and checks equality against ground-truth answers.
  - `code_execution_reward`: Executes generated Python snippets within a timeout constraint (extracted from the Sudoku RL notebook reference).

### 4.5. Training Runners (`src/slm_post_train/trainers/`)
- **`sft_runner.py`**:
  - Coordinates model loading, data preparation, TRL `SFTTrainer` with `SFTConfig`.
  - Executes training and prints peak reserved VRAM memory metrics.
  - Saves trained LoRA adapters to the specified output directory.
- **`grpo_runner.py`**:
  - Coordinates model loading, prompt dataset setup, and reward resolution.
  - Configures TRL `GRPOTrainer` with `GRPOConfig` (tailored for single GPU / 8GB VRAM).
  - Executes policy optimization and saves adapters.

### 4.6. Model Export (`src/slm_post_train/export/exporter.py`)
Provides three export pathways:
1. **LoRA Adapter**: Standard PEFT adapter weights (`save_pretrained`).
2. **Merged Checkpoint**: Full weights merged in 16-bit or 4-bit (`save_pretrained_merged`).
3. **GGUF Export**: Quantized binary export (`save_pretrained_gguf`) supporting `f16`, `q4_k_m`, and `q8_0` for Ollama/llama.cpp.

### 4.7. Dummy Data & Smoke Tests
- `data/dummy/sft_sample.jsonl`: Minimal multi-turn instruction dataset.
- `data/dummy/grpo_sample.jsonl`: Minimal reasoning dataset with ground truth.
- `configs/sft/smoke_test.yaml` and `configs/grpo/smoke_test.yaml`: Minimal configurations (`max_steps: 2`, `batch_size: 1`) allowing instant verification.

## 5. Extensibility to Vision-Language Models (VLMs)
- The model loader will inspect the `modality` field (`"text"` vs `"vision"`).
- Multimodal data collators can be added to `src/slm_post_train/data/` under a dedicated `vision_data.py` module when VLM support is activated.
- Training loops remain decoupled from modality-specific tokenization.

## 6. Error Handling & Quality Control
- Graceful validation on missing dataset files or missing keys in YAML configs.
- Time-bounded safe execution for code-based rewards to avoid infinite loops or hung processes.
- Clear error reporting if CUDA is unavailable or if memory limits are exceeded.
