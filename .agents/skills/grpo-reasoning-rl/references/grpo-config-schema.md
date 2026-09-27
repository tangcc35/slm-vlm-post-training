# GRPO YAML Configuration Schema & Reference

Comprehensive reference for configuring Group Relative Policy Optimization (GRPO) reinforcement learning jobs in `slm-post-train`.

---

## Table of Contents
1. [Overview](#overview)
2. [Root Configuration Fields](#root-configuration-fields)
3. [Section `model`](#section-model)
4. [Section `lora`](#section-lora)
5. [Section `dataset`](#section-dataset)
6. [Section `rewards`](#section-rewards)
7. [Section `training`](#section-training)
8. [Section `output`](#section-output)
9. [Complete Annotated Example](#complete-annotated-example)
10. [Key Hyperparameters Deep Dive](#key-hyperparameters-deep-dive)
    - [`num_generations`](#num_generations)
    - [`max_completion_length`](#max_completion_length)
    - [`beta` (KL Penalty)](#beta-kl-penalty)
    - [`temperature`](#temperature)
11. [Monitoring via `GRPOMonitorCallback`](#monitoring-via-grpomonitorcallback)
12. [Hardware Memory Budgeting (<= 8GB VRAM)](#hardware-memory-budgeting--8gb-vram)

---

## Overview

The GRPO training pipeline executes online reinforcement learning where policy rollouts are sampled on the fly, evaluated via custom reward functions, and optimized using relative group advantages without requiring a separate critic network.

Jobs are initiated via the CLI:
```bash
uv run slm-post-train train --config configs/grpo/your_config.yaml
```

The runner loads the base model in 4-bit via Unsloth, applies LoRA, formats prompt-only datasets, resolves reward functions from the registry, and initializes TRL's `GRPOTrainer` with `GRPOConfig`.

---

## Root Configuration Fields

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `stage` | `string` | **Yes** | `"grpo"` | Pipeline stage. Must be set to `"grpo"` for reinforcement learning. |
| `model` | `dict` | **Yes** | `{}` | Base model checkpoint, context length, and quantization settings. |
| `lora` | `dict` | No | `{}` | LoRA adapter rank, alpha, and target projection layers. |
| `dataset` | `dict` | **Yes** | `{}` | Prompt dataset location, split, and sample limits. |
| `rewards` | `list[str]` | **Yes** | `["xml_format"]` | Names of registered reward functions to evaluate completions. |
| `training` | `dict` | No | `{}` | GRPO rollout parameters, optimization hyper-parameters, and schedules. |
| `output` | `dict` | No | `{}` | Checkpoint output directory and logging frequency. |

---

## Section `model`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name_or_path` | `string` | `""` | Hugging Face model ID (e.g. `"unsloth/gemma-2-2b-it"`, `"unsloth/Qwen2.5-1.5B-Instruct"`) or local path. |
| `max_seq_length` | `integer` | `512` | Total token window (`max_prompt_length + max_completion_length`). Keep <= 1024 on 8GB VRAM. |
| `load_in_4bit` | `boolean` | `true` | Enables bitsandbytes 4-bit quantization for base model weights. Essential for low VRAM. |
| `dtype` | `string` or `null` | `null` | Precision override (`"bfloat16"`, `"float16"`). Defaults to auto-detection based on GPU support. |
| `modality` | `string` | `"text"` | Modality identifier (`"text"`). |

---

## Section `lora`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `r` | `integer` | `16` | Rank dimension for LoRA adaptation matrices (typically 8, 16, or 32). |
| `lora_alpha` | `integer` | `32` | Scaling factor. Conventionally set to `2 * r`. |
| `target_modules` | `list[str]` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Linear layers to adapt. Use attention projections for minimal VRAM; add MLP projections (`["gate_proj", "up_proj", "down_proj"]`) for higher capacity. |
| `lora_dropout` | `float` | `0.0` | LoRA dropout. Kept at 0.0 for Unsloth fast kernel optimization. |
| `bias` | `string` | `"none"` | Bias tuning setting (`"none"` recommended). |
| `random_state` | `integer` | `3407` | Random seed for initialization. |
| `use_gradient_checkpointing` | `string` / `bool` | `"unsloth"` | Activation checkpointing. Saves significant activation memory during policy updates. |

---

## Section `dataset`

GRPO datasets must contain prompts and optional ground truth metadata. **Never include assistant completions in the prompt text.**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `string` | `"data/dummy/grpo_sample.jsonl"` | Path to `.jsonl`, `.json`, `.csv` file, or HF dataset ID. |
| `split` | `string` | `"train"` | Dataset split name. |
| `max_samples` | `integer` or `null` | `null` | Maximum samples to load. Useful for smoke testing. |

The dataset loader automatically recognizes input columns:
- Prompt: looks for `prompt`, `question`, or `problem`. Formatted as `[{"role": "user", "content": ...}]`.
- Metadata: looks for `answer`, `solution`, or `ground_truth`. Forwarded to reward functions via `kwargs`.

---

## Section `rewards`

A list of string identifiers corresponding to functions decorated with `@register_reward`:

```yaml
rewards:
  - "xml_format"
  - "exact_match"
  - "code_execution"
```

Each reward function evaluates all completions for a prompt group and produces a list of floats. The total reward for a rollout is the unweighted sum of individual reward values (unless normalized or shaped within custom functions).

---

## Section `training`

Controls rollout generation, policy loss calculation, and optimizer steps in `GRPOConfig`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_generations` | `integer` | `2` | Number of rollouts sampled per prompt ($G$). Must be $\ge 2$ to compute group relative advantages. Recommended 2–4 for 8GB VRAM. |
| `max_prompt_length` | `integer` | `256` | Maximum token length for the input question prompt. |
| `max_completion_length` | `integer` | `256` | Maximum token length for the policy rollout response. |
| `learning_rate` | `float` | `5.0e-6` | Policy peak learning rate. Typically 5x to 10x smaller than SFT (e.g. `5e-6` to `1e-5`). |
| `beta` | `float` | `0.04` | KL divergence penalty coefficient ($\beta$) between the active policy and reference policy. |
| `temperature` | `float` | `0.8` | Sampling temperature for rollouts. Controls generation diversity within the group. |
| `batch_size` | `integer` | `1` | Prompts per device per training step. |
| `gradient_accumulation_steps` | `integer` | `2` | Number of update steps accumulated before optimizer step. |
| `max_steps` | `integer` | `10` | Maximum training steps. Smoke test: 2; full run: 50–200. |
| `optim` | `string` | `"paged_adamw_8bit"` | Optimizer choice. Paged 8-bit AdamW prevents OOM spikes. |
| `use_vllm` | `boolean` | `false` | Enable vLLM rollout acceleration. Requires vLLM backend installed. |
| `warmup_ratio` | `float` | `0.1` | Fraction of total steps for learning rate warmup. |
| `lr_scheduler_type` | `string` | `"cosine"` | Learning rate schedule (`"cosine"`, `"linear"`). |
| `weight_decay` | `float` | `0.1` | Weight decay regularization. |

---

## Section `output`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `output_dir` | `string` | `"outputs/grpo_model"` | Directory to save final LoRA adapter weights and tokenizer. |
| `logging_steps` | `integer` | `1` | Frequency of training metric logging. |
| `report_to` | `string` | `"none"` | Reporting dashboard (`"none"`, `"wandb"`). |

---

## Complete Annotated Example

Below is the production configuration for Sudoku reasoning RL (`configs/grpo/gemma_sudoku_rl.yaml`):

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
  num_generations: 4
  max_steps: 50
  max_prompt_length: 256
  max_completion_length: 512
  learning_rate: 5.0e-6

output:
  output_dir: "outputs/gemma_sudoku_grpo"
  logging_steps: 2
```

---

## Key Hyperparameters Deep Dive

### `num_generations`
- **Definition**: The group size $G$ sampled for each prompt during the rollout phase.
- **Why it matters**: GRPO calculates advantage by standardizing rewards *within* this group:
  $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$
- **Tuning rule**:
  - $G=2$: Minimum viable group size. Lowest memory footprint, but higher gradient variance.
  - $G=4$: Optimal balance for single-GPU setups (<= 8GB VRAM). Provides sufficient diversity while fitting within VRAM limits.
  - $G \ge 8$: Recommended for multi-GPU or >= 24GB VRAM setups. Yields tighter advantage estimates.

### `max_completion_length`
- **Definition**: Maximum new tokens generated per rollout response.
- **Impact on VRAM**: Rollouts must be stored in KV memory and later computed through forward/backward passes. A rollout length of 1024 with $G=4$ consumes $4 \times 1024 = 4096$ completion tokens per prompt.
- **Recommendation**: For <= 8GB VRAM, cap `max_completion_length` at 256 or 512.

### `beta` (KL Penalty)
- **Definition**: Weight $\beta$ multiplying the KL divergence penalty $\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$.
- **Dynamics**:
  - **$\beta$ too low ($< 0.001$)**: The policy can collapse, outputting repetitive gibberish or extreme syntax hacks that fool verifiers while destroying natural language capability.
  - **$\beta$ too high ($> 0.1$)**: The policy remains tethered to the reference model, refusing to explore novel reasoning chains or learn new strategies.
  - **Standard value**: $0.01$ to $0.04$.

### `temperature`
- **Definition**: Sampling temperature during rollout generation.
- **Critical rule**: Must be $> 0$ (typically $0.7 - 0.9$).
- **The Zero-Variance Trap**: If temperature is set to 0.0 (greedy decoding), all $G$ completions for a prompt will be identical. When all completions are identical, their rewards are identical, standard deviation is zero, and advantage collapses to 0. The policy receives zero gradient update and learning stalls completely.

---

## Monitoring via `GRPOMonitorCallback`

To prevent silent training failures and detect reward hacking early, implement or attach a monitoring callback during GRPO runs:

```python
from transformers import TrainerCallback

class GRPOMonitorCallback(TrainerCallback):
    """Monitors reward distribution, KL divergence, and completion statistics during GRPO."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Logged metrics include:
        # - reward/{func_name}: Mean score per reward verifier
        # - kl: Estimated KL divergence between policy and reference model
        # - completion_length: Average tokens generated per rollout
        # - advantage_std: Standard deviation of advantages within groups
        step = state.global_step
        kl = logs.get("kl", logs.get("objective/kl", 0.0))
        if kl > 0.5:
            print(f"[WARNING] Step {step}: KL divergence ({kl:.4f}) is elevated. Consider increasing beta.")
```

### Metrics Diagnostic Table

| Symptom | Diagnostic | Recommended Fix |
| :--- | :--- | :--- |
| **Reward increases, but KL skyrockets (> 0.5)** | Policy is drifting drastically from base model; potential language collapse. | Increase `beta` (e.g. from 0.02 to 0.05). |
| **All reward scores flat at 0.0** | Task too difficult or prompt format unrecognized; zero positive reinforcement. | Add dense shaping reward (e.g., partial format credit) or pre-train with SFT first. |
| **Completion length steadily increases to `max_completion_length`** | Verbosity reward hacking; model learns longer chains artificially increase score. | Add length penalty or tighten `max_completion_length`. |
| **Zero advantage std across steps** | Temperature too low or model generating identical rollouts. | Increase `temperature` to 0.8–1.0. |

---

## Hardware Memory Budgeting (<= 8GB VRAM)

GRPO has distinct memory dynamics compared to SFT:
1. **Rollout Generation**: Generates $G \times \text{batch\_size}$ completions.
2. **Forward/Backward Evaluation**: Computes log probabilities under active policy $\pi_\theta$ and reference model $\pi_{\text{ref}}$.
3. **Reference Model Memory**: In Unsloth QLoRA, the base 4-bit model serves as the reference model (by disabling LoRA adapters during evaluation), avoiding a second model copy in memory.

### 8GB VRAM Recommended Settings

```yaml
model:
  name_or_path: "unsloth/gemma-2-2b-it"
  max_seq_length: 512
  load_in_4bit: true

lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  use_gradient_checkpointing: "unsloth"

training:
  batch_size: 1
  gradient_accumulation_steps: 2
  num_generations: 2
  max_prompt_length: 128
  max_completion_length: 256
  optim: "paged_adamw_8bit"
  learning_rate: 5.0e-6
```

Estimated Memory Breakdown on Gemma 2B:
- **Base 4-bit Model**: ~1.6 GB
- **LoRA Adapter Weights**: ~0.05 GB
- **Optimizer States (paged 8-bit)**: ~0.4 GB
- **Rollout KV Cache & Activations ($G=2$, len=256)**: ~2.5–3.2 GB
- **Total Peak VRAM**: **~4.8–5.5 GB** (comfortably within 8GB limit).
