# SFT YAML Configuration Schema & Reference

Comprehensive reference for configuring Supervised Fine-Tuning (SFT) jobs in `slm-post-train`.

---

## Table of Contents
1. [Overview](#overview)
2. [Root Configuration Fields](#root-configuration-fields)
3. [Section `model`](#section-model)
4. [Section `lora`](#section-lora)
5. [Section `dataset`](#section-dataset)
6. [Section `training`](#section-training)
7. [Section `output`](#section-output)
8. [Complete Annotated Example](#complete-annotated-example)
9. [Hardware Memory Budgeting (<=8GB VRAM)](#hardware-memory-budgeting--8gb-vram)

---

## Overview

The SFT training pipeline consumes a declarative YAML configuration file passed via the CLI:

```bash
uv run slm-post-train train --config configs/sft/your_config.yaml
```

The runner initializes the model via Unsloth's `FastLanguageModel` (or `FastVisionModel`), applies LoRA parameter-efficient adapters, formats the dataset using HuggingFace chat templates, attaches the TRL `SFTTrainer`, and masks prompt tokens with `unsloth.chat_templates.train_on_responses_only`.

---

## Root Configuration Fields

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `stage` | `string` | **Yes** | `"sft"` | Identifies the pipeline stage. Must be set to `"sft"` for supervised instruction training. |
| `model` | `dict` | **Yes** | `{}` | Base model identifier, context window, and quantization parameters. |
| `lora` | `dict` | No | `{}` | PEFT/LoRA adapter topology and hyperparameters. |
| `dataset` | `dict` | **Yes** | `{}` | Training data source, chat template, and sample splits. |
| `training` | `dict` | No | `{}` | Optimization hyper-parameters, batch size, and scheduler rules. |
| `output` | `dict` | No | `{}` | Output paths and logging parameters. |

---

## Section `model`

Configures the base model architecture and precision loading.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name_or_path` | `string` | `""` | Hugging Face repo ID (e.g., `"unsloth/gemma-2-2b-it"`, `"unsloth/Qwen2.5-7B-Instruct"`) or local filesystem directory. |
| `max_seq_length` | `integer` | `2048` | Maximum sequence length (tokens). Keep <= 2048 for 8GB VRAM limits; can increase if memory allows. |
| `load_in_4bit` | `boolean` | `true` | Enables bitsandbytes 4-bit NF4 quantization. Critical for low-VRAM training. |
| `dtype` | `string` or `null` | `null` | Tensor precision: `"bfloat16"`, `"float16"`, `"float32"`, or `null` (auto-detected based on hardware Ampere+ support). |
| `modality` | `string` | `"text"` | Modality type: `"text"` loads `FastLanguageModel`; `"vision"` loads `FastVisionModel`. |

---

## Section `lora`

Configures low-rank adaptation matrices attached to linear layers.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `r` | `integer` | `16` | Rank dimension for LoRA projection matrices. Common values: 8, 16, 32, 64. |
| `lora_alpha` | `integer` | `2 * r` | Scaling factor. Standard practice is `2 * r` (or 16). |
| `target_modules` | `list[str]` | All 7 projections | Linear projection layers to adapt: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. For minimal memory, use attention only: `["q_proj", "k_proj", "v_proj", "o_proj"]`. |
| `lora_dropout` | `float` | `0.0` | Dropout probability for LoRA layers. Kept at `0.0` to utilize Unsloth fast CUDA kernels. |
| `bias` | `string` | `"none"` | Bias training strategy: `"none"`, `"all"`, or `"lora_only"`. `"none"` recommended. |
| `random_state` | `integer` | `3407` | Seed for initializing LoRA weight adapters. |
| `use_gradient_checkpointing` | `string` / `bool` | `"unsloth"` | Activation checkpointing method. Set to `"unsloth"` to save up to 30% additional VRAM. |

---

## Section `dataset`

Specifies the dataset source, split, and conversational formatting.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `string` | `"data/dummy/sft_sample.jsonl"` | File path (`.jsonl`, `.json`, `.csv`) or HuggingFace dataset name (e.g. `"mlabonne/FineTome-100k"`). |
| `chat_template` | `string` | `"chatml"` | Chat template identifier. Supported: `chatml`, `llama-3`, `gemma`, `phi-3`, `mistral`, `alpaca`, `vicuna`, etc. |
| `split` | `string` | `"train"` | Dataset split or slice syntax (e.g. `"train"`, `"train[:3000]"`). |
| `max_samples` | `integer` or `null` | `null` | Optional upper bound on training examples for smoke testing or rapid iteration. |
| `dataset_num_proc` | `integer` | `2` | Number of worker processes used by `datasets.map` for preprocessing. |
| `instruction_part` | `string` or `null` | `null` | Explicit instruction delimiter override for responses-only loss masking. |
| `response_part` | `string` or `null` | `null` | Explicit response delimiter override for responses-only loss masking. |

---

## Section `training`

Hyperparameters controlling the optimization loop via TRL's `SFTConfig`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `batch_size` | `integer` | `1` | Per-device training batch size. Set to 1 or 2 for <=8GB VRAM. |
| `gradient_accumulation_steps` | `integer` | `2` | Steps to accumulate gradients before executing an optimizer step. Effective batch size = `batch_size * gradient_accumulation_steps`. |
| `learning_rate` | `float` | `2.0e-4` | Peak learning rate for LoRA adapters. Typical range: `1e-4` to `3e-4`. |
| `max_steps` | `integer` | `60` | Total optimization steps to train. Set to `1`-`2` for smoke tests, `100`-`500` for task fine-tuning. |
| `warmup_steps` | `integer` | `5` | Steps for linear warmup of learning rate. |
| `optim` | `string` | `"adamw_8bit"` | Optimizer implementation. Use `"adamw_8bit"` or `"paged_adamw_8bit"` to reduce optimizer state footprint. |
| `weight_decay` | `float` | `0.01` | L2 weight decay regularization. |
| `lr_scheduler_type` | `string` | `"linear"` | Learning rate schedule: `"linear"`, `"cosine"`, `"constant"`. |
| `seed` | `integer` | `3407` | Random seed for data shuffling and training reproducibility. |
| `train_on_responses_only` | `boolean` | `true` | Masks instruction/prompt tokens with `-100` so loss is computed solely on assistant output tokens. |
| `packing` | `boolean` | `false` | Concatenates sequences up to `max_seq_length`. **Warning:** Must be `false` when `train_on_responses_only` is `true`. |

---

## Section `output`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `output_dir` | `string` | `"outputs/sft_model"` | Directory where trained LoRA adapter weights and tokenizer configs are saved. |
| `logging_steps` | `integer` | `1` | Frequency (in steps) to log training loss and metrics to standard out and WandB. |

---

## Complete Annotated Example

Below is the standard production configuration for fine-tuning Gemma 2 2B (`configs/sft/gemma_text_sft.yaml`):

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
  train_on_responses_only: true

output:
  output_dir: "outputs/gemma_sft"
  logging_steps: 5
```

---

## Hardware Memory Budgeting (<=8GB VRAM)

When operating under tight VRAM constraints (e.g., RTX 3070 / RTX 4060 / laptop 8GB VRAM GPUs):

1. **4-Bit NF4 Quantization (`load_in_4bit: true`)**:
   - Reduces a 2B parameter model base weights from ~4.5 GB (fp16) to ~1.4 GB.
   - Reduces a 7B parameter model base weights from ~14 GB (fp16) to ~4.5 GB.
2. **8-Bit Optimizers (`optim: "adamw_8bit"` or `"paged_adamw_8bit"`)**:
   - 32-bit AdamW stores 8 bytes per trainable parameter (first and second moments).
   - 8-bit AdamW cuts optimizer state memory by 75%.
   - `"paged_adamw_8bit"` pages out excess states to CPU RAM during activation spikes.
3. **Activation Checkpointing (`use_gradient_checkpointing: "unsloth"`)**:
   - Discards intermediate forward activations and recalculates them during backward pass.
   - Unsloth's optimized CUDA kernels cut checkpointing overhead while eliminating OOM spikes.
4. **Target Modules**:
   - 7 projection modules (`q, k, v, o, gate, up, down`) provide full expressive capacity.
   - If sequence length is extended beyond 2048 or batch size is increased, reduce `target_modules` to attention projections only: `["q_proj", "k_proj", "v_proj", "o_proj"]`.
5. **Sequence Length**:
   - KV cache and self-attention matrices scale quadratically with sequence length $O(N^2)$.
   - For 8GB VRAM, cap `max_seq_length` at 2048 tokens. For smoke testing or quick verifications, use 512 tokens.
6. **Batch Sizing**:
   - Always set `batch_size: 1` or `2`.
   - Scale effective batch size using `gradient_accumulation_steps` (e.g., 4 or 8) to maintain optimization stability without VRAM penalty.
