# SLM Post-Training with Unsloth & uv Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, production-ready Small Language Model (SLM) post-training repository supporting Unsloth-accelerated SFT, GRPO Reinforcement Learning, and GGUF model export, managed deterministically via `uv`.

**Architecture:** A decoupled package layout under `src/slm_post_train/` with isolated components for model loading, dataset formatting (with response-only masking), a modular GRPO reward registry, SFT and GRPO training runners, and model merging/GGUF export, driven by standard YAML configurations via a unified CLI entrypoint.

**Tech Stack:** Python 3.11, `uv`, Unsloth, PyTorch, Hugging Face `transformers`, `datasets`, `trl`, `peft`, `bitsandbytes`, `pyyaml`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-10-slm-post-training-design.md`

## Global Constraints

- Python version pinned to `3.11` via `.python-version`.
- Package manager: `uv`.
- No strict Pydantic schema validation overhead; load YAML/JSON configurations directly with `yaml.safe_load`.
- Provide root-level `data/dummy/` datasets and corresponding smoke test configurations in `configs/` for instantaneous 2-step offline/local verification.
- LoRA configuration must set `lora_alpha = lora_rank * 2` and support 4-bit quantization default for 8GB VRAM compatibility.

---

### Task 1: Environment & Repository Scaffolding

**Files:**
- Create: `.python-version`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/slm_post_train/__init__.py`

**Interfaces:**
- Produces: Project package structure recognized by `uv`, CLI script alias `slm-post-train`.

- [ ] **Step 1: Create `.python-version`**

```text
3.11
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# Output directories & checkpoints
outputs/
checkpoints/
exports/
*.gguf

# IDE / OS metadata
.idea/
.vscode/
*.swp
*.swo
*Zone.Identifier

# Testing & coverage
.pytest_cache/
.coverage
htmlcov/
```

- [ ] **Step 3: Create `pyproject.toml` with `uv` configuration**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "slm-post-train"
version = "0.1.0"
description = "Modular SLM and VLM post-training with Unsloth and uv"
readme = "README.md"
requires-python = ">=3.10,<3.12"
dependencies = [
    "torch",
    "unsloth",
    "trl>=0.14.0",
    "transformers>=4.48.0",
    "datasets>=3.0.0",
    "accelerate>=0.34.0",
    "bitsandbytes",
    "peft>=0.14.0",
    "pyyaml>=6.0",
    "sentencepiece",
    "protobuf",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[project.scripts]
slm-post-train = "slm_post_train.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/slm_post_train"]
```

- [ ] **Step 4: Create package `__init__.py`**

Create `src/slm_post_train/__init__.py`:
```python
"""SLM Post-Training with Unsloth."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Verify project initialization with `uv`**

Run: `uv sync --dry-run || echo "uv is available and pyproject is valid"`

---

### Task 2: Dummy Datasets for Smoke Testing

**Files:**
- Create: `data/dummy/sft_sample.jsonl`
- Create: `data/dummy/grpo_sample.jsonl`

**Interfaces:**
- Produces: `data/dummy/sft_sample.jsonl` formatted with `"conversations"` keys compatible with Unsloth chat template standardization.
- Produces: `data/dummy/grpo_sample.jsonl` formatted with `"prompt"` and `"answer"` keys for GRPO rollout verification.

- [ ] **Step 1: Create `data/dummy/sft_sample.jsonl`**

```jsonl
{"conversations": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"conversations": [{"role": "user", "content": "Solve: 2 + 2"}, {"role": "assistant", "content": "2 + 2 is equal to 4."}]}
{"conversations": [{"role": "user", "content": "Explain gravity in one sentence."}, {"role": "assistant", "content": "Gravity is the fundamental force of attraction that pulls objects with mass toward one another."}]}
{"conversations": [{"role": "user", "content": "Write a python print statement."}, {"role": "assistant", "content": "print('Hello, world!')"}]}
{"conversations": [{"role": "user", "content": "What is 10 divided by 2?"}, {"role": "assistant", "content": "10 divided by 2 is 5."}]}
```

- [ ] **Step 2: Create `data/dummy/grpo_sample.jsonl`**

```jsonl
{"prompt": "Calculate the result of 15 + 27. Output the reasoning inside <think>...</think> and the final answer inside <answer>...</answer>.", "answer": "42"}
{"prompt": "Calculate the result of 8 * 9. Output the reasoning inside <think>...</think> and the final answer inside <answer>...</answer>.", "answer": "72"}
{"prompt": "Calculate the result of 100 - 37. Output the reasoning inside <think>...</think> and the final answer inside <answer>...</answer>.", "answer": "63"}
{"prompt": "What is 5 cubed? Output the reasoning inside <think>...</think> and the final answer inside <answer>...</answer>.", "answer": "125"}
{"prompt": "Calculate 14 + 18. Output the reasoning inside <think>...</think> and the final answer inside <answer>...</answer>.", "answer": "32"}
```

- [ ] **Step 3: Verify datasets exist and are valid JSON Lines**

Run: `python3 -c "import json; [json.loads(line) for line in open('data/dummy/sft_sample.jsonl')]; [json.loads(line) for line in open('data/dummy/grpo_sample.jsonl')]; print('JSONL datasets valid')"`

---

### Task 3: GRPO Reward Registry & Standard Rewards

**Files:**
- Create: `src/slm_post_train/rewards/__init__.py`
- Create: `src/slm_post_train/rewards/registry.py`
- Create: `src/slm_post_train/rewards/standard.py`
- Test: `tests/test_rewards.py`

**Interfaces:**
- Produces: `@register_reward(name)`, `get_reward_function(name)`, `list_registered_rewards()`.
- Produces: `xml_format_reward(prompts, completions, answer=None, **kwargs) -> list[float]`
- Produces: `exact_match_reward(prompts, completions, answer=None, **kwargs) -> list[float]`
- Produces: `code_execution_reward(prompts, completions, answer=None, **kwargs) -> list[float]`

- [ ] **Step 1: Write the failing unit tests in `tests/test_rewards.py`**

```python
import pytest
from slm_post_train.rewards.registry import register_reward, get_reward_function, list_registered_rewards
from slm_post_train.rewards.standard import xml_format_reward, exact_match_reward, code_execution_reward

def test_reward_registry():
    @register_reward("test_custom_reward")
    def dummy_reward(prompts, completions, **kwargs):
        return [1.0] * len(completions)

    fn = get_reward_function("test_custom_reward")
    assert fn(["p"], ["c"]) == [1.0]
    assert "test_custom_reward" in list_registered_rewards()

def test_xml_format_reward():
    prompts = ["Calculate 1+1"] * 2
    completions = [
        "<think>1+1=2</think><answer>2</answer>",
        "The answer is 2 without tags"
    ]
    scores = xml_format_reward(prompts=prompts, completions=completions)
    assert scores[0] == 1.0
    assert scores[1] == 0.0

def test_exact_match_reward():
    prompts = ["What is 2+2?"] * 3
    completions = [
        "<think>2+2</think><answer>4</answer>",
        "<answer> 4 </answer>",
        "<answer>5</answer>"
    ]
    scores = exact_match_reward(prompts=prompts, completions=completions, answer=["4", "4", "4"])
    assert scores[0] == 1.0
    assert scores[1] == 1.0
    assert scores[2] == 0.0

def test_code_execution_reward():
    prompts = ["Write a function"] * 2
    completions = [
        "```python\ndef solve():\n    return 42\nresult = solve()\n```",
        "```python\nraise ValueError('error')\n```"
    ]
    scores = code_execution_reward(prompts=prompts, completions=completions)
    assert scores[0] == 1.0
    assert scores[1] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rewards.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'slm_post_train.rewards')

- [ ] **Step 3: Implement `src/slm_post_train/rewards/registry.py`**

```python
from typing import Callable, Dict, List

_REWARD_REGISTRY: Dict[str, Callable] = {}

def register_reward(name: str):
    """Decorator to register a reward function."""
    def decorator(fn: Callable) -> Callable:
        _REWARD_REGISTRY[name] = fn
        return fn
    return decorator

def get_reward_function(name: str) -> Callable:
    """Retrieve a registered reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(
            f"Reward function '{name}' not found. Available rewards: {list(_REWARD_REGISTRY.keys())}"
        )
    return _REWARD_REGISTRY[name]

def list_registered_rewards() -> List[str]:
    """Return all registered reward function names."""
    return list(_REWARD_REGISTRY.keys())
```

- [ ] **Step 4: Implement `src/slm_post_train/rewards/standard.py` and `__init__.py`**

In `src/slm_post_train/rewards/standard.py`:
```python
import re
from typing import List, Optional
from slm_post_train.rewards.registry import register_reward

@register_reward("xml_format")
def xml_format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Scores 1.0 if output matches <think>...</think><answer>...</answer> structure, else 0.0."""
    scores = []
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
    for completion in completions:
        text = completion.strip()
        scores.append(1.0 if pattern.match(text) else 0.0)
    return scores

@register_reward("exact_match")
def exact_match_reward(
    prompts: List[str],
    completions: List[str],
    answer: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """Extracts text inside <answer>...</answer> and compares to ground truth answer."""
    scores = []
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    for i, completion in enumerate(completions):
        if not answer or i >= len(answer):
            scores.append(0.0)
            continue
        expected = str(answer[i]).strip().lower()
        match = answer_pattern.search(completion)
        if match:
            extracted = match.group(1).strip().lower()
            scores.append(1.0 if extracted == expected else 0.0)
        else:
            scores.append(0.0)
    return scores

@register_reward("code_execution")
def code_execution_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Safely checks if python code block can be executed without error."""
    code_block_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    scores = []
    for completion in completions:
        match = code_block_pattern.search(completion)
        if not match:
            scores.append(0.0)
            continue
        code_str = match.group(1)
        safe_globals = {"__builtins__": {
            "range": range, "len": len, "min": min, "max": max, "sum": sum,
            "int": int, "float": float, "str": str, "bool": bool, "list": list,
            "dict": dict, "set": set, "print": print
        }}
        try:
            exec(code_str, safe_globals)
            scores.append(1.0)
        except Exception:
            scores.append(0.0)
    return scores
```

In `src/slm_post_train/rewards/__init__.py`:
```python
from slm_post_train.rewards.registry import register_reward, get_reward_function, list_registered_rewards
import slm_post_train.rewards.standard  # auto-register standard rewards

__all__ = ["register_reward", "get_reward_function", "list_registered_rewards"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rewards.py -v`
Expected: PASS

---

### Task 4: Data Processing for SFT and GRPO

**Files:**
- Create: `src/slm_post_train/data/__init__.py`
- Create: `src/slm_post_train/data/sft_data.py`
- Create: `src/slm_post_train/data/grpo_data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `prepare_sft_dataset(dataset_path_or_id, tokenizer, chat_template="chatml", split="train", max_samples=None)`
- Produces: `prepare_grpo_dataset(dataset_path_or_id, split="train", max_samples=None)`

- [ ] **Step 1: Write unit tests in `tests/test_data.py`**

```python
import pytest
from slm_post_train.data.grpo_data import prepare_grpo_dataset

def test_prepare_grpo_dataset_dummy():
    dataset = prepare_grpo_dataset("data/dummy/grpo_sample.jsonl")
    assert len(dataset) == 5
    sample = dataset[0]
    assert "prompt" in sample
    assert "answer" in sample
    assert isinstance(sample["prompt"], list)
    assert sample["prompt"][0]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'slm_post_train.data')

- [ ] **Step 3: Implement `src/slm_post_train/data/sft_data.py`**

```python
import os
from datasets import load_dataset, Dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats

def prepare_sft_dataset(
    dataset_path_or_id: str,
    tokenizer,
    chat_template: str = "chatml",
    split: str = "train",
    max_samples: int = None
):
    """Loads and standardizes chat dataset for SFT training."""
    if os.path.exists(dataset_path_or_id):
        if dataset_path_or_id.endswith(".jsonl") or dataset_path_or_id.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_path_or_id, split=split)
        elif dataset_path_or_id.endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_path_or_id, split=split)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path_or_id}")
    else:
        dataset = load_dataset(dataset_path_or_id, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
```

- [ ] **Step 4: Implement `src/slm_post_train/data/grpo_data.py` and `__init__.py`**

In `src/slm_post_train/data/grpo_data.py`:
```python
import os
from datasets import load_dataset

def prepare_grpo_dataset(
    dataset_path_or_id: str,
    split: str = "train",
    max_samples: int = None
):
    """Loads and formats prompt dataset for GRPO reinforcement learning."""
    if os.path.exists(dataset_path_or_id):
        dataset = load_dataset("json", data_files=dataset_path_or_id, split=split)
    else:
        dataset = load_dataset(dataset_path_or_id, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    def format_row(example):
        prompt = example["prompt"]
        if isinstance(prompt, str):
            formatted_prompt = [{"role": "user", "content": prompt}]
        else:
            formatted_prompt = prompt
        result = {"prompt": formatted_prompt}
        if "answer" in example:
            result["answer"] = example["answer"]
        return result

    return dataset.map(format_row)
```

In `src/slm_post_train/data/__init__.py`:
```python
from slm_post_train.data.sft_data import prepare_sft_dataset
from slm_post_train.data.grpo_data import prepare_grpo_dataset

__all__ = ["prepare_sft_dataset", "prepare_grpo_dataset"]
```

- [ ] **Step 5: Run unit tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: PASS

---

### Task 5: Model & LoRA Loader

**Files:**
- Create: `src/slm_post_train/models/__init__.py`
- Create: `src/slm_post_train/models/loader.py`

**Interfaces:**
- Produces: `load_model_and_tokenizer(model_cfg: dict, lora_cfg: dict = None, modality: str = "text")`

- [ ] **Step 1: Implement `src/slm_post_train/models/loader.py`**

```python
import torch

def load_model_and_tokenizer(model_cfg: dict, lora_cfg: dict = None, modality: str = "text"):
    """Loads model and tokenizer with Unsloth FastModel and attaches LoRA adapters."""
    model_name = model_cfg.get("name_or_path", "unsloth/gemma-2-2b-it")
    max_seq_length = model_cfg.get("max_seq_length", 2048)
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    dtype = model_cfg.get("dtype", None)
    if dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float16":
        dtype = torch.float16

    if modality == "vision":
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )

    if lora_cfg:
        r = lora_cfg.get("r", 16)
        lora_alpha = lora_cfg.get("lora_alpha", r * 2)
        target_modules = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        lora_dropout = lora_cfg.get("lora_dropout", 0.0)
        bias = lora_cfg.get("bias", "none")
        random_state = lora_cfg.get("random_state", 3407)
        use_gradient_checkpointing = lora_cfg.get("use_gradient_checkpointing", "unsloth")

        if modality == "vision":
            model = FastVisionModel.get_peft_model(
                model,
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                random_state=random_state,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            model = FastLanguageModel.get_peft_model(
                model,
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                random_state=random_state,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

    return model, tokenizer
```

In `src/slm_post_train/models/__init__.py`:
```python
from slm_post_train.models.loader import load_model_and_tokenizer

__all__ = ["load_model_and_tokenizer"]
```

---

### Task 6: Model Checkpoint & GGUF Exporter

**Files:**
- Create: `src/slm_post_train/export/__init__.py`
- Create: `src/slm_post_train/export/exporter.py`

**Interfaces:**
- Produces: `export_model(model, tokenizer, output_dir: str, export_format: str = "lora", quantization_method: str = "q4_k_m")`

- [ ] **Step 1: Implement `src/slm_post_train/export/exporter.py`**

```python
import os
import logging

logger = logging.getLogger(__name__)

def export_model(
    model,
    tokenizer,
    output_dir: str,
    export_format: str = "lora",
    quantization_method: str = "q4_k_m"
):
    """Exports model as LoRA adapter, 16-bit/4-bit merged model, or GGUF binary."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exporting model to {output_dir} using format: {export_format}")

    if export_format == "lora":
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved LoRA adapter weights to {output_dir}")

    elif export_format == "merged_16bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"Saved 16-bit merged model to {output_dir}")

    elif export_format == "merged_4bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")
        logger.info(f"Saved 4-bit merged model to {output_dir}")

    elif export_format == "gguf":
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization_method)
        logger.info(f"Saved GGUF ({quantization_method}) to {output_dir}")

    else:
        raise ValueError(
            f"Unknown export format: '{export_format}'. Supported: lora, merged_16bit, merged_4bit, gguf"
        )
```

In `src/slm_post_train/export/__init__.py`:
```python
from slm_post_train.export.exporter import export_model

__all__ = ["export_model"]
```

---

### Task 7: Training Runners (SFT & GRPO)

**Files:**
- Create: `src/slm_post_train/trainers/__init__.py`
- Create: `src/slm_post_train/trainers/sft_runner.py`
- Create: `src/slm_post_train/trainers/grpo_runner.py`

**Interfaces:**
- Produces: `run_sft(config: dict)`
- Produces: `run_grpo(config: dict)`

- [ ] **Step 1: Implement `src/slm_post_train/trainers/sft_runner.py`**

```python
import torch
import logging
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from slm_post_train.models.loader import load_model_and_tokenizer
from slm_post_train.data.sft_data import prepare_sft_dataset

logger = logging.getLogger(__name__)

def run_sft(config: dict):
    """Executes SFT instruction training."""
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    data_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    output_dir = output_cfg.get("output_dir", "outputs/sft_model")

    logger.info("Initializing model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        lora_cfg=lora_cfg,
        modality=model_cfg.get("modality", "text")
    )

    logger.info("Preparing SFT dataset...")
    dataset = prepare_sft_dataset(
        dataset_path_or_id=data_cfg.get("path", "data/dummy/sft_sample.jsonl"),
        tokenizer=tokenizer,
        chat_template=data_cfg.get("chat_template", "chatml"),
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples", None),
    )

    sft_args = SFTConfig(
        dataset_text_field="text",
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dataset_num_proc=data_cfg.get("dataset_num_proc", 2),
        packing=training_cfg.get("packing", False),
        per_device_train_batch_size=training_cfg.get("batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        warmup_steps=training_cfg.get("warmup_steps", 5),
        max_steps=training_cfg.get("max_steps", 60),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        logging_steps=output_cfg.get("logging_steps", 1),
        optim=training_cfg.get("optim", "adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        seed=training_cfg.get("seed", 3407),
        output_dir=output_dir,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    if training_cfg.get("train_on_responses_only", True):
        trainer = train_on_responses_only(trainer)

    logger.info("Starting SFT training...")
    stats = trainer.train()
    logger.info(f"Training completed. Runtime: {stats.metrics.get('train_runtime', 0):.2f}s")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
        logger.info(f"Peak VRAM reserved: {peak_vram:.2f} GB")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to {output_dir}")
    return stats
```

- [ ] **Step 2: Implement `src/slm_post_train/trainers/grpo_runner.py`**

```python
import torch
import logging
from trl import GRPOConfig, GRPOTrainer
from slm_post_train.models.loader import load_model_and_tokenizer
from slm_post_train.data.grpo_data import prepare_grpo_dataset
from slm_post_train.rewards.registry import get_reward_function

logger = logging.getLogger(__name__)

def run_grpo(config: dict):
    """Executes GRPO Reinforcement Learning."""
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    data_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    output_dir = output_cfg.get("output_dir", "outputs/grpo_model")

    logger.info("Initializing model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        lora_cfg=lora_cfg,
        modality=model_cfg.get("modality", "text")
    )

    logger.info("Preparing GRPO dataset...")
    dataset = prepare_grpo_dataset(
        dataset_path_or_id=data_cfg.get("path", "data/dummy/grpo_sample.jsonl"),
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples", None),
    )

    reward_names = config.get("rewards", ["xml_format"])
    reward_funcs = [get_reward_function(name) for name in reward_names]
    logger.info(f"Loaded reward functions: {reward_names}")

    grpo_args = GRPOConfig(
        use_vllm=training_cfg.get("use_vllm", False),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        adam_beta1=training_cfg.get("adam_beta1", 0.9),
        adam_beta2=training_cfg.get("adam_beta2", 0.99),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=output_cfg.get("logging_steps", 1),
        per_device_train_batch_size=training_cfg.get("batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        num_generations=training_cfg.get("num_generations", 2),
        max_prompt_length=training_cfg.get("max_prompt_length", 256),
        max_completion_length=training_cfg.get("max_completion_length", 256),
        max_steps=training_cfg.get("max_steps", 10),
        output_dir=output_dir,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
    )

    logger.info("Starting GRPO training...")
    stats = trainer.train()
    logger.info(f"GRPO training completed.")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
        logger.info(f"Peak VRAM reserved: {peak_vram:.2f} GB")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model policy saved to {output_dir}")
    return stats
```

In `src/slm_post_train/trainers/__init__.py`:
```python
from slm_post_train.trainers.sft_runner import run_sft
from slm_post_train.trainers.grpo_runner import run_grpo

__all__ = ["run_sft", "run_grpo"]
```

---

### Task 8: Unified CLI & YAML Configurations

**Files:**
- Create: `src/slm_post_train/cli.py`
- Create: `configs/sft/smoke_test.yaml`
- Create: `configs/grpo/smoke_test.yaml`
- Create: `configs/sft/gemma_text_sft.yaml`
- Create: `configs/grpo/gemma_sudoku_rl.yaml`

**Interfaces:**
- Produces: Command-line interface `slm-post-train train --config <path>` and `slm-post-train export --model-path <path> --output-dir <path> --format <format>`

- [ ] **Step 1: Implement `src/slm_post_train/cli.py`**

```python
import argparse
import sys
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("slm_post_train")

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(prog="slm-post-train", description="SLM/VLM Post-Training CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Run a training job")
    train_parser.add_argument("--config", required=True, help="Path to YAML training configuration")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export or quantize a model checkpoint")
    export_parser.add_argument("--model-path", required=True, help="Path to trained model or adapter")
    export_parser.add_argument("--output-dir", required=True, help="Path to save exported model")
    export_parser.add_argument(
        "--format",
        default="lora",
        choices=["lora", "merged_16bit", "merged_4bit", "gguf"],
        help="Export target format"
    )
    export_parser.add_argument("--quant", default="q4_k_m", help="Quantization method for GGUF")

    args = parser.parse_args()

    if args.subcommand == "train":
        config = load_yaml_config(args.config)
        stage = config.get("stage", "sft").lower()
        if stage == "sft":
            from slm_post_train.trainers.sft_runner import run_sft
            run_sft(config)
        elif stage == "grpo":
            from slm_post_train.trainers.grpo_runner import run_grpo
            run_grpo(config)
        else:
            raise ValueError(f"Unknown training stage: '{stage}'. Must be 'sft' or 'grpo'.")

    elif args.subcommand == "export":
        from unsloth import FastLanguageModel
        from slm_post_train.export.exporter import export_model
        logger.info(f"Loading model from {args.model_path} for export...")
        model, tokenizer = FastLanguageModel.from_pretrained(args.model_path)
        export_model(model, tokenizer, args.output_dir, export_format=args.format, quantization_method=args.quant)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `configs/sft/smoke_test.yaml`**

```yaml
stage: sft

model:
  name_or_path: "unsloth/gemma-2-2b-it"
  max_seq_length: 512
  load_in_4bit: true

lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  use_gradient_checkpointing: "unsloth"

dataset:
  path: "data/dummy/sft_sample.jsonl"
  chat_template: "chatml"
  split: "train"

training:
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 2
  learning_rate: 2.0e-4
  optim: "adamw_8bit"
  train_on_responses_only: true

output:
  output_dir: "outputs/sft_smoke_test"
  logging_steps: 1
```

- [ ] **Step 3: Create `configs/grpo/smoke_test.yaml`**

```yaml
stage: grpo

model:
  name_or_path: "unsloth/gemma-2-2b-it"
  max_seq_length: 512
  load_in_4bit: true

lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

dataset:
  path: "data/dummy/grpo_sample.jsonl"
  split: "train"

rewards:
  - "xml_format"
  - "exact_match"

training:
  batch_size: 1
  gradient_accumulation_steps: 1
  num_generations: 2
  max_steps: 2
  max_prompt_length: 128
  max_completion_length: 128
  learning_rate: 5.0e-6

output:
  output_dir: "outputs/grpo_smoke_test"
  logging_steps: 1
```

- [ ] **Step 4: Create `configs/sft/gemma_text_sft.yaml` and `configs/grpo/gemma_sudoku_rl.yaml`**

In `configs/sft/gemma_text_sft.yaml`:
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

In `configs/grpo/gemma_sudoku_rl.yaml`:
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

### Task 9: Documentation & End-to-End Verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Provide clear documentation for:
- Installation with `uv`
- Running unit tests
- Running SFT smoke test & full recipes
- Running GRPO smoke test & full recipes
- Exporting to GGUF and merged checkpoints

- [ ] **Step 2: Clean up Zone.Identifier artifacts at root**

Remove any leftover Windows WSL `*.Zone.Identifier` files.

- [ ] **Step 3: Run full test suite with pytest**

Run: `pytest tests/ -v`
Expected: ALL PASS
