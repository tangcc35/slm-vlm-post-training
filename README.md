# SLM & VLM Post-Training & Story/RP Engine

Modular, production-ready framework for post-training Small Language Models (SLMs) and Vision-Language Models (VLMs) using **Unsloth**, **Hugging Face TRL**, **PyTorch**, and **uv**, paired with a high-performance **Dual-Mode Story & Roleplay Engine** powered by **Google ADK (Agent Development Kit 2.0)** and **FastAPI**.

Designed from the ground up to train on consumer GPUs with **≤ 8GB VRAM** via 4-bit QLoRA, memory-efficient Unsloth kernels, and paged 8-bit optimizers, and deploy into agentic narrative workflows.

---

## Key Features

### Post-Training & Export (SLM / VLM)
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

### Dual-Mode Story & Roleplay Engine (Google ADK 2.0)
- **Story Co-Pilot Workflow:** Graph-based multi-agent pipeline where a **Story Director** plans pacing and tone guidance and a **Story Writer** crafts seamless literary continuations.
- **Character Card V2 Roleplay:** Native character card parsing, `{{char}}`/`{{user}}` macro interpolation, dynamic keyword-triggered lorebook retrieval via ADK `before_model_callback`, and author's note steering.
- **Context Compaction (Google ADK):** Hybrid token-based safety net and sliding-window turn compaction (`EventsCompactionConfig` + `LlmEventSummarizer`) for automatic context compression across both single RP agents and multi-agent story workflows, keeping context footprint bounded while preserving recent raw interactions.
- **Production-Ready FastAPI Server:** Server-Sent Events (SSE) streaming with configurable chunk buffering, database session persistence (`DatabaseSessionService` with SQLite/aiosqlite), and multi-provider LLM support via **LiteLLM**.
- **Modern Python Tooling:** Fast, reproducible environments managed via `uv` and Python 3.11.

---

## Repository Architecture

```text
slm-vlm-post-training/
├── .agents/
│   └── skills/                      # Agent skills framework (agentskills.io standard)
│       ├── unsloth-sft/             # SFT training, conversational datasets & QLoRA
│       ├── grpo-reasoning-rl/       # GRPO RL, XML format, reward functions & KL tuning
│       ├── model-export-gguf/       # LoRA merging, GGUF quantization & Ollama serving
│       ├── character-rp/            # Character Card V2, lorebook & macro interpolation
│       ├── story-copilot/           # ADK 2.0 Director/Writer workflow & narrative graph
│       ├── story-rp-backend/        # FastAPI routes, SSE streaming, SQLite & Phoenix
│       └── story-rp-frontend/       # Dual-mode UI, SSE readers, character modals
├── configs/
│   ├── sft/
│   │   ├── smoke_test.yaml          # Fast 2-step SFT sanity check
│   │   ├── gemma_text_sft.yaml      # Gemma 2B SFT recipe (FineTome-100k)
│   │   └── qwen35_08b_nsfw_story.yaml # Qwen 3.5 0.8B story fine-tuning recipe
│   └── grpo/
│       ├── smoke_test.yaml          # Fast 2-step GRPO sanity check
│       └── gemma_sudoku_rl.yaml     # Gemma 2B Sudoku reasoning GRPO recipe
├── data/
│   └── dummy/
│       ├── sft_sample.jsonl         # Dummy dataset for SFT testing
│       └── grpo_sample.jsonl        # Dummy dataset for GRPO testing
├── src/
│   ├── slm_post_train/              # Post-training framework (SFT, GRPO, export)
│   │   ├── cli.py                   # Unified CLI entrypoint (train, export)
│   │   ├── data/                    # Conversational SFT & GRPO dataset loaders
│   │   ├── export/                  # LoRA, merged 16/4-bit, and GGUF exporter
│   │   ├── models/                  # FastLanguageModel / FastVisionModel loader
│   │   ├── rewards/                 # Extensible reward decorator & registry
│   │   └── trainers/                # SFT & GRPO Unsloth training runners
│   └── story_rp_engine/             # Dual-Mode Story & Roleplay Engine (ADK 2.0)
│       ├── api/                     # FastAPI app, roleplay & story routes, SSE streaming
│       ├── core/                    # EngineConfig, types, LiteLLM provider, AgentRegistry
│       ├── rp/                      # Character Card V2 parser, prompt builder, lorebook engine
│       ├── story/                   # Director & Writer agents, declarative ADK workflow graph
│       └── storage/                 # ADK DatabaseSessionService & local JSON store
├── tests/
│   ├── story_rp_engine/             # 90+ tests for Character Card V2, lorebook, graph, API
│   └── ...                          # 100+ tests for SFT, GRPO, rewards, configs, datasets
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
- Dual-mode Story & Roleplay Engine (Character Card V2, lorebook keyword retrieval, ADK workflow graph, SSE streaming, SQLite persistence).
- Agent skills framework validation (frontmatter schema, progressive disclosure references, `openai.yaml` metadata).

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

## Dual-Mode Story & Roleplay Engine (ADK 2.0)

The `story_rp_engine` package provides a high-performance agentic engine built on **Google ADK (Agent Development Kit 2.0)**, **LiteLLM**, and **FastAPI**. It features two specialized interaction modes:
1. **Interactive Roleplay (`/api/v1/rp/*`):** Character Card V2 compliant roleplay agent with dynamic keyword-triggered lorebook / world-info injection, author's note steering, and persistent multi-turn session storage.
2. **Story Co-Pilot (`/api/v1/story/*`):** Multi-agent narrative collaboration where a **Story Director** plans tone, pacing, and emotional focus, and a **Story Writer** produces literary prose continuations.

### Architecture & Agent Flow

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Roleplay Mode (Single-Agent)                    │
│                                                                        │
│   User Prompt + Author's Note                                          │
│        │                                                               │
│        ▼                                                               │
│   [rp_before_model_callback] ──► Keyword scan & Lorebook injection     │
│        │                                                               │
│        ▼                                                               │
│   [ADK LlmAgent (Character Card V2)] ──► Streaming SSE / Turn Reply   │
│        │                                                               │
│        ▼                                                               │
│   [ADK DatabaseSessionService (SQLite / aiosqlite)]                    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                      Story Co-Pilot Mode (Multi-Agent)                 │
│                                                                        │
│   Story Premise + Existing Text + User Instruction                     │
│        │                                                               │
│        ▼                                                               │
│   [ADK Workflow Graph]                                                 │
│        │                                                               │
│        ├──► [Story Director Agent] (Scene framing, tone, pacing advice)│
│        │          │                                                    │
│        └──► [Story Writer Agent] (Literary prose completion)            │
│                   │                                                    │
│                   ▼                                                    │
│        Streaming SSE / Full Expansion Reply                            │
└────────────────────────────────────────────────────────────────────────┘
```

### Configuration & Environment Variables

Configure model backends, endpoints, and storage through environment variables or `EngineConfig`:

| Variable | Default | Description |
|---|---|---|
| `STORY_RP_MODEL` | `ollama/llama3.1:8b` | Model identifier parsed by LiteLLM (e.g., `openai/...`, `ollama/...`, `gemini/...`). |
| `STORY_RP_API_BASE` | None | Base URL for custom endpoints (e.g., `http://localhost:11434`, `http://localhost:8080/v1`). |
| `STORY_RP_API_KEY` | None | API key if required (automatically defaults to `"local"` for local OpenAI-compatible endpoints). |
| `STORY_RP_TEMPERATURE` | `0.8` | Generation sampling temperature (`0.0` - `2.0`). |
| `STORY_RP_TOP_P` | `0.9` | Top-p nucleus sampling cutoff (`0.0` - `1.0`). |
| `STORY_RP_MAX_TOKENS` | `131072` | Maximum generation tokens buffer. |
| `STORY_RP_STORAGE_DIR` | `.engine_data` | Directory path for persisted character cards, lorebooks, and sessions. |
| `STORY_RP_DB_URL` | SQLite in storage dir | SQLAlchemy database URL for ADK session persistence (e.g., `sqlite+aiosqlite:///...`). |
| `STORY_RP_SKIP_WARMUP` | `0` | Set to `1` or `true` to skip model inference warmup on startup. |
| `STORY_RP_COMPACTION_ENABLED` | `1` | Enable ADK context compaction (`1`/`true`/`yes` to enable, `0`/`false` to disable). |
| `STORY_RP_COMPACTION_TOKEN_THRESHOLD` | `4000` | Token limit triggering pre-invocation tail-retention context compaction. |
| `STORY_RP_COMPACTION_EVENT_RETENTION_SIZE` | `5` | Number of recent raw events kept intact when token compaction triggers. |
| `STORY_RP_COMPACTION_INTERVAL` | `10` | Number of conversational turns between sliding-window compactions. |
| `STORY_RP_COMPACTION_OVERLAP_SIZE` | `2` | Number of prior turns to retain as overlapping context in sliding-window summaries. |
| `STORY_RP_COMPACTION_PROMPT_TEMPLATE` | None | Optional custom summarizer prompt template containing `{conversation_history}`. |

### Running the API Server

Start the FastAPI application with Uvicorn:

```bash
# Option 1: Using default local Ollama model (llama3.1:8b)
uv run uvicorn story_rp_engine.api.app:create_app --factory --host 0.0.0.0 --port 8000

# Option 2: Using custom fine-tuned model served with llama-cpp-python or vLLM
export STORY_RP_MODEL="openai/custom-slm"
export STORY_RP_API_BASE="http://localhost:8080/v1"
uv run uvicorn story_rp_engine.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Verify service status:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### API Endpoints Reference

#### 1. Character Management (`Character Card V2`)

Store and retrieve Character Card V2 JSON definitions with `{{char}}` and `{{user}}` macro interpolation:

- **Create / Update Character:**
  ```bash
  curl -X POST "http://localhost:8000/api/v1/characters" \
      -H "Content-Type: application/json" \
      -d '{
        "char_id": "elena",
        "name": "Elena",
        "description": "An intrepid archaeologist exploring ancient ruins.",
        "personality": "Curious, witty, and cautious.",
        "scenario": "Elena is trapped in a hidden chamber with {{user}}.",
        "first_mes": "Watch your step! These pressure plates are ancient.",
        "mes_example": "<START>\n{{user}}: What is that?\n{{char}}: An artifact older than the dynasty.",
        "system_prompt": "Stay in character as Elena.",
        "post_history_instructions": "Keep answers descriptive."
      }'
  ```

- **List Characters:**
  ```bash
  curl http://localhost:8000/api/v1/characters
  ```

- **Get Character:**
  ```bash
  curl http://localhost:8000/api/v1/characters/elena
  ```

#### 2. Roleplay Mode

- **Standard Chat (`POST /api/v1/rp/chat`):**
  ```bash
  curl -X POST http://localhost:8000/api/v1/rp/chat \
      -H "Content-Type: application/json" \
      -d '{
        "char_id": "elena",
        "session_id": "session_101",
        "user_name": "Explorer",
        "message": "Did you hear that sound down the hall?",
        "authors_note": "A distant rumble echoes from the floor above."
      }'
  ```

- **Streaming SSE Chat (`POST /api/v1/rp/chat/stream`):**
  Stream token deltas in real-time with configurable chunk buffering:
  ```bash
  curl -N -X POST http://localhost:8000/api/v1/rp/chat/stream \
      -H "Content-Type: application/json" \
      -d '{
        "char_id": "elena",
        "session_id": "session_101",
        "message": "What should we do next?",
        "chunk_size": 4
      }'
  ```

#### 3. Story Co-Pilot Mode

Collaborative narrative generation using the declarative two-agent pipeline (`story_director` $\rightarrow$ `story_writer`):

- **Expand Story (`POST /api/v1/story/expand`):**
  ```bash
  curl -X POST http://localhost:8000/api/v1/story/expand \
      -H "Content-Type: application/json" \
      -d '{
        "session_id": "story_session_101",
        "premise": "Two detectives uncover a hidden underground vault beneath a vintage clock tower.",
        "genre": "Mystery / Noir",
        "tone": "Suspenseful and atmospheric",
        "current_text": "Rain lashed against the stained glass windows. Detective Vance held the brass key in his hand.",
        "instruction": "Describe the moment Vance inserts the key and the heavy gear mechanism begins to turn."
      }'
  ```

- **Streaming SSE Expansion (`POST /api/v1/story/expand/stream`):**
  ```bash
  curl -N -X POST http://localhost:8000/api/v1/story/expand/stream \
      -H "Content-Type: application/json" \
      -d '{
        "session_id": "story_session_101",
        "premise": "Two detectives uncover a hidden underground vault beneath a vintage clock tower.",
        "genre": "Mystery / Noir",
        "tone": "Suspenseful and atmospheric",
        "current_text": "Rain lashed against the stained glass windows. Detective Vance held the brass key in his hand.",
        "instruction": "Describe the moment Vance inserts the key and the heavy gear mechanism begins to turn.",
        "chunk_size": 4
      }'
  ```

#### 4. Session History, Compaction & Turn Management

- **Get Session Turns & Compaction Summaries (`GET /api/v1/rp/sessions/{session_id}/turns`):**
  Inspect conversation turns, including automatically generated context compaction summaries (`role: "compaction"`, `is_compaction: true`):
  ```bash
  curl http://localhost:8000/api/v1/rp/sessions/session_101/turns
  # Response:
  # {
  #   "turns": [
  #     {"index": 0, "role": "compaction", "text": "Compacted conversation summary...", "is_compaction": true},
  #     {"index": 1, "role": "user", "text": "What should we do next?"},
  #     {"index": 2, "role": "model", "text": "Let us check the corridor."}
  #   ]
  # }
  ```

- **Delete Single Turn or Rewind History (`POST /api/v1/rp/sessions/{session_id}/turns/delete`):**
  Atomically delete a specific turn or truncate/rewind all turns subsequent to an index:
  ```bash
  curl -X POST http://localhost:8000/api/v1/rp/sessions/session_101/turns/delete \
      -H "Content-Type: application/json" \
      -d '{
        "turn_index": 2,
        "truncate_subsequent": false
      }'
  ```

- **Clear Entire Session (`DELETE /api/v1/rp/sessions/{session_id}`):**
  ```bash
  curl -X DELETE http://localhost:8000/api/v1/rp/sessions/session_101
  ```

### Connecting Fine-Tuned SLMs to the Story & RP Engine

Fine-tuned models exported with `slm_post_train` can be served locally and plugged directly into `story_rp_engine`:

1. **Train & Export Model:**
   ```bash
   # Train SFT model (e.g. Qwen story recipe)
   uv run slm-post-train train --config configs/sft/qwen35_08b_nsfw_story.yaml

   # Export to GGUF format
   uv run slm-post-train export \
       --model-path outputs/qwen35_08b_nsfw_story \
       --output-dir exports/story_slm_gguf \
       --format gguf \
       --quant q4_k_m
   ```

2. **Serve Locally:**
   ```bash
   # With llama-cpp-python server
   uv run python -m llama_cpp.server \
       --model exports/story_slm_gguf/unsloth.Q4_K_M.gguf \
       --port 8080
   ```

3. **Launch Engine connected to local server:**
   ```bash
   export STORY_RP_MODEL="openai/story-slm"
   export STORY_RP_API_BASE="http://localhost:8080/v1"
   uv run uvicorn story_rp_engine.api.app:create_app --factory --port 8000
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

## Agent Skills Framework

This repository provides an automated, domain-specific **Agent Skills Framework** following the [agentskills.io](https://agentskills.io) standard, located in [`.agents/skills/`](.agents/skills/). AI coding agents (Antigravity, Claude Code, Cursor, OpenAI Codex) automatically discover and activate these modular skills to maintain architectural consistency, safety constraints, and idiomatic patterns across both the post-training pipeline and the story/RP engine.

### Progressive Disclosure Architecture

To maximize context-window efficiency and token economics, the skill framework adheres to a 3-tier progressive disclosure model:

1. **Discovery (Level 1 - System Prompt):** Lightweight YAML frontmatter (`name`, `description`) in `SKILL.md` is registered on agent startup (~50–100 tokens per skill), allowing the agent to know *when* to invoke each skill without consuming context.
2. **Workflow & Instructions (Level 2 - Task Context):** When an agent encounters a relevant domain task, it reads `SKILL.md` (~1,000–2,000 tokens), which contains step-by-step procedures, rules, decision trees, and verification checklists.
3. **Deep References (Level 3 - On-Demand):** High-density technical references, API schemas, and code recipes in `references/*.md` (~2,000–5,000 tokens) are loaded strictly when needed for deep debugging or complex implementations.

Each skill also includes an `agents/openai.yaml` definition for seamless multi-platform compatibility across OpenAI and Codex tooling.

### Available Skills

| Skill | Directory | Core Capabilities |
|---|---|---|
| **unsloth-sft** | [`.agents/skills/unsloth-sft/`](.agents/skills/unsloth-sft/) | SFT instruction fine-tuning, responses-only loss masking, ShareGPT/HF conversational formats, 4-bit QLoRA memory budgeting (≤8GB VRAM), and training YAML recipe configuration. |
| **grpo-reasoning-rl** | [`.agents/skills/grpo-reasoning-rl/`](.agents/skills/grpo-reasoning-rl/) | Group Relative Policy Optimization (GRPO), prompt-only dataset formatting, custom `@register_reward` scoring functions (XML format, exact match, code execution), and KL divergence monitoring. |
| **model-export-gguf** | [`.agents/skills/model-export-gguf/`](.agents/skills/model-export-gguf/) | Standalone LoRA adapter export, 16-bit / 4-bit merged checkpoints, GGUF quantization (`q4_k_m`, `q8_0`, `f16`), Ollama `Modelfile` generation, and local `llama.cpp` inference testing. |
| **character-rp** | [`.agents/skills/character-rp/`](.agents/skills/character-rp/) | Character Card V2 JSON specification, `{{char}}`/`{{user}}` macro interpolation, dynamic keyword-triggered lorebook retrieval via ADK `before_model_callback`, and Author's Note steering. |
| **story-copilot** | [`.agents/skills/story-copilot/`](.agents/skills/story-copilot/) | Google ADK 2.0 declarative multi-agent workflow graph, Story Director planning agent, Story Writer prose generation agent, session state tracking, and narrative pacing guidance. |
| **story-rp-backend** | [`.agents/skills/story-rp-backend/`](.agents/skills/story-rp-backend/) | FastAPI REST endpoints, Server-Sent Events (SSE) streaming with chunk buffering, SQLite `DatabaseSessionService`, LiteLLM provider routing, and Phoenix OpenTelemetry tracing. |
| **story-rp-frontend** | [`.agents/skills/story-rp-frontend/`](.agents/skills/story-rp-frontend/) | Vanilla ES6+ dual-mode web interface (`src/story_rp_engine/web/`), SSE client streaming readers (`fetch` + `ReadableStream`), Markdown rendering, Character Card JSON import modal, and responsive layout. |

### Verifying Agent Skills

The framework includes automated pytest validation enforcing frontmatter validity, directory structure, reference link resolution, and `openai.yaml` schema compliance:

```bash
uv run pytest tests/test_agent_skills.py -v
```

---

## Development & Contributing

### Formatting & Linting
Ensure code style consistency before submitting pull requests:
```bash
uv run pytest tests/ -v
```

### License
Apache 2.0. See [LICENSE](LICENSE) for details.
