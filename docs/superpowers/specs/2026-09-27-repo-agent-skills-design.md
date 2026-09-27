# Repository Agent Skills Design Specification

- **Date:** 2026-09-27
- **Status:** Approved
- **Target Directory:** `.agents/skills/`
- **Scope:** 7 Modular Agent Skills covering SLM/VLM Post-Training & Export and the Dual-Mode Story/RP Engine

---

## 1. Executive Summary & Goals

This specification defines the architecture, directory layout, progressive disclosure patterns, and verification standards for repository-level AI agent skills in `slm-vlm-post-training`.

The design is built upon:
1. **The Agent Skills Open Standard ([agentskills.io](https://agentskills.io/specification))**: Self-contained skill directories utilizing YAML frontmatter and progressive disclosure.
2. **Anthropic Official Skill Authoring Best Practices**: Token-budget efficiency, Skill Discovery Optimization (SDO), 1-level-deep reference linking, and calibrated degrees of freedom.
3. **Hugging Face's *Training Agents* Reference ([burtenshaw/training-agents](https://github.com/burtenshaw/training-agents/tree/main/.agents/skills))**: Operational method ladders, verifiable rewards, and cross-platform compatibility with Codex/OpenAI (`agents/openai.yaml`), Claude Code, Google Antigravity, and GitHub Copilot.

---

## 2. Universal Standards & Guardrails

### 2.1 Directory Structure
Each skill adheres to the canonical layout:

```text
.agents/skills/<skill-name>/
├── SKILL.md                 # Primary entrypoint (Metadata + Stepwise instructions)
├── agents/
│   └── openai.yaml          # Codex / OpenAI agent integration metadata
└── references/              # Focused, 1-level-deep reference docs (loaded on demand)
    ├── <topic-1>.md
    └── <topic-2>.md
```

### 2.2 Skill Discovery Optimization (SDO)
- **`name`**: Lowercase alphanumeric and hyphens only, 1–64 characters.
- **`description`**: Must start with `"Use when..."` and state **triggering conditions, symptoms, and technologies** in the third person. It must **never** summarize the workflow or recipe (which causes agents to bypass reading the full `SKILL.md`).
- Target length: Under 500 characters.

### 2.3 Progressive Disclosure & Token Budgeting
- **Startup Indexing**: Agents load only the frontmatter `name` and `description` (~100 tokens per skill).
- **Activation**: When triggered, agents load `SKILL.md` (< 500 lines).
- **Deep Reference**: Detailed schemas, tables, and lengthy guides live in `references/` and are read on demand.
- **One-Level Depth**: No chaining (`SKILL.md` links to `references/*.md`; reference files do not link to further nested references). Any reference file longer than 100 lines must include a Table of Contents at the top.

### 2.4 Calibrated Degrees of Freedom & Verification Gates
- **Low Freedom**: Hardware constraints (≤ 8 GB VRAM, `paged_adamw_8bit`, `unsloth` gradient checkpointing), responses-only loss masking, and SSE streaming formats.
- **Medium Freedom**: Prompt macros, character card attributes, and story pacing heuristics.
- **Mandatory Verification**: Every skill defines an explicit, non-destructive smoke-test command that agents must execute before claiming task completion.

---

## 3. The 7 Skills Specifications

### 3.1 `unsloth-sft` (Supervised Fine-Tuning)
- **Purpose**: Supervised fine-tuning of SLMs and VLMs using Unsloth, 4-bit QLoRA, responses-only loss masking, and conversational datasets.
- **Trigger Description**: `Use when designing, configuring, running, or debugging Supervised Fine-Tuning (SFT) for SLMs or VLMs with Unsloth, including responses-only loss masking, conversational ShareGPT/HF message formatting, 4-bit QLoRA, memory budgeting for <=8GB VRAM, or training YAML configs.`
- **Key Guardrails**:
  - Always mask prompt tokens via `unsloth.chat_templates.train_on_responses_only`.
  - Always use `paged_adamw_8bit` and `gradient_checkpointing: "unsloth"` on <= 8 GB VRAM GPUs.
  - Checkpoints and large datasets must remain outside version control (`outputs/` and `data/` ignored).
- **Verification Gate**:
  ```bash
  uv run slm-post-train train-sft --config configs/sft/smoke_test.yaml --smoke-test
  ```
- **References**:
  - `references/sft-config-schema.md`: YAML configuration schema, LoRA target modules, and learning rate schedules.
  - `references/dataset-formatting.md`: ShareGPT/HuggingFace conversational formats, role tagging, and response masking patterns.

### 3.2 `grpo-reasoning-rl` (Group Relative Policy Optimization)
- **Purpose**: Online reinforcement learning for reasoning models using Unsloth + TRL `GRPOTrainer`, multi-completion rollouts, and custom reward functions.
- **Trigger Description**: `Use when designing, implementing, configuring, or debugging Group Relative Policy Optimization (GRPO) reinforcement learning for reasoning SLMs, including prompt-only dataset formatting, custom @register_reward functions, XML structure verification, code execution verifiers, or KL divergence monitoring.`
- **Key Guardrails**:
  - Training dataset must be prompt-only (no assistant completions in input).
  - Rewards must be registered via `@register_reward` and remain deterministic where possible.
  - Always monitor reward distributions and KL drift via `GRPOMonitorCallback` to prevent reward hacking.
- **Verification Gate**:
  ```bash
  uv run slm-post-train train-grpo --config configs/grpo/smoke_test.yaml --smoke-test
  ```
- **References**:
  - `references/reward-functions.md`: Implementing rewards with `@register_reward`, XML parsing, and safe code execution.
  - `references/grpo-config-schema.md`: Hyperparameters (`num_generations`, `max_completion_length`, `beta`, `temperature`).

### 3.3 `model-export-gguf` (Adapter Merging & Quantization)
- **Purpose**: Merging trained LoRA adapters and exporting models to 16-bit, 4-bit, or quantized GGUF binaries for local deployment.
- **Trigger Description**: `Use when exporting, merging, quantizing, or verifying trained SLM/VLM models, including saving 16-bit or 4-bit merged checkpoints, converting to GGUF format (q4_k_m, q8_0, f16), serving via llama.cpp or Ollama, or testing local inference.`
- **Key Guardrails**:
  - Ensure disk space (> 15 GB free for full merges).
  - Verify exported GGUF with `scripts/infer_gguf.py` before deploying to production servers.
- **Verification Gate**:
  ```bash
  uv run python -c "from slm_post_train.export.exporter import ModelExporter; print('Exporter loaded successfully')"
  ```
- **References**:
  - `references/export-workflows.md`: CLI commands and Python APIs for `merged_16bit`, `merged_4bit`, and GGUF quantization.
  - `references/llama-cpp-serving.md`: Setting up `llama-server` via `run_sh/serve_llama_cpp.sh` and Ollama Modelfiles.

### 3.4 `character-rp` (Roleplay & Lorebook Engine)
- **Purpose**: Character Card V2 parsing, dynamic macro interpolation, keyword-triggered lorebook retrieval via Google ADK callbacks, and Author's Note steering.
- **Trigger Description**: `Use when creating, editing, debugging, or testing Character Card V2 JSON definitions, dynamic lorebook keyword matching, {{char}}/{{user}} prompt macro substitution, Author's Note injection, or Google ADK RP agent callbacks.`
- **Key Guardrails**:
  - Character Card V2 JSON must conform to standard specification (`spec: "chara_card_v2"`).
  - Lorebook keyword matching must prevent prompt bloat through token budget caps and selective recursion.
  - Macros must interpolate cleanly across first messages, alternate greetings, and system prompts.
- **Verification Gate**:
  ```bash
  uv run pytest tests/story_rp_engine/test_character.py tests/story_rp_engine/test_lorebook.py
  ```
- **References**:
  - `references/character-card-v2-spec.md`: Full JSON schema, fields (`name`, `description`, `personality`, `scenario`, `mes_example`).
  - `references/lorebook-matching-rules.md`: Keyword matching rules, regex support, activation logic, and token limits.

### 3.5 `story-copilot` (Director & Writer ADK Workflow)
- **Purpose**: Collaborative multi-agent storytelling pipeline with Google ADK 2.0, coordinating Story Director (pacing, tone, outline) and Story Writer (prose continuation).
- **Trigger Description**: `Use when designing, configuring, or modifying the Google ADK 2.0 Story Co-Pilot workflow, including Story Director planning, Story Writer prose generation, narrative workflow graphs, scene pacing guidance, or story branching states.`
- **Key Guardrails**:
  - Clear agent separation: Director analyzes narrative state and outputs structured guidance; Writer only consumes guidance and context to draft literary prose.
  - State transitions must be atomic and reversible for branching storylines.
- **Verification Gate**:
  ```bash
  uv run pytest tests/story_rp_engine/test_story_workflow.py
  ```
- **References**:
  - `references/director-writer-pipeline.md`: Agent prompts, guidance structures, and collaborative flow.
  - `references/story-workflow-graph.md`: Declarative ADK graph definition, state transitions, and node routing.

### 3.6 `story-rp-backend` (FastAPI, SSE Streaming & Persistence)
- **Purpose**: Backend REST APIs, Server-Sent Events (SSE) streaming protocols, SQLite/aiosqlite session persistence, LiteLLM routing, and Phoenix observability.
- **Trigger Description**: `Use when developing, debugging, or extending the FastAPI backend for the Story and RP engine, including REST routes, Server-Sent Events (SSE) streaming, SQLite session storage via DatabaseSessionService, LiteLLM model provider routing, or Phoenix OpenTelemetry tracing.`
- **Key Guardrails**:
  - Streaming endpoints must output valid SSE events (`event: message`, `data: ...`, `event: done`).
  - All database queries must be asynchronous (`aiosqlite`).
  - Graceful fallback for multi-provider LLM timeouts via LiteLLM.
- **Verification Gate**:
  ```bash
  uv run pytest tests/story_rp_engine/test_api.py
  ```
- **References**:
  - `references/api-endpoints-and-sse.md`: Route definitions, request/response schemas, and SSE chunk buffering.
  - `references/database-session-storage.md`: SQLite table schemas, session lifecycle, and migration patterns.

### 3.7 `story-rp-frontend` (Web Interface & Dual-Mode UI)
- **Purpose**: Single-page browser interface in `src/story_rp_engine/web/`, handling dual-mode switching (RP Chat vs Story Co-Pilot), client-side SSE streaming, and card/lorebook management.
- **Trigger Description**: `Use when creating, modifying, or styling the browser interface for the Story and RP engine in src/story_rp_engine/web/, including dual-mode UI switching, client-side SSE streaming readers, markdown rendering, character card import modals, or CSS layout improvements.`
- **Key Guardrails**:
  - Maintain lightweight vanilla HTML5/CSS3/ES6 (no heavy npm build chains required).
  - Robust SSE error handling (reconnect or display friendly error if backend disconnects).
  - Responsive layout for desktop and mobile viewports.
- **Verification Gate**:
  ```bash
  uv run python -c "import os; assert os.path.exists('src/story_rp_engine/web/index.html') and os.path.exists('src/story_rp_engine/web/app.js')"
  ```
- **References**:
  - `references/ui-components-and-modes.md`: DOM element mappings, mode switching logic, and styling variables.
  - `references/sse-event-client.md`: JavaScript `EventSource` / `fetch` reader implementation for chunked markdown streams.

---

## 4. Implementation Steps

1. Create directory tree under `.agents/skills/` for all 7 skills.
2. Author `agents/openai.yaml` for each skill to enable Codex/OpenAI workspace interoperability.
3. Author `references/*.md` with tables of contents, schemas, and runnable code blocks.
4. Author `SKILL.md` for each skill with SDO-compliant YAML frontmatter and concise operational instructions.
5. Run the verification test suite to ensure all referenced files, test commands, and configurations are valid.
