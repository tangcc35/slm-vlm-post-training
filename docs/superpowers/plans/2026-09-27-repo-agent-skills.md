# Repository Agent Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a production-grade, SOTA-compliant `.agents/skills/` repository skills framework for `slm-vlm-post-training` containing 7 modular skills across post-training and the story/RP engine.

**Architecture:** Each skill is a self-contained directory with `SKILL.md` (agentskills.io YAML frontmatter and concise progressive disclosure workflow), `agents/openai.yaml` (Codex/OpenAI workspace compatibility), and `references/*.md` (1-level-deep modular references). A pytest validation suite enforces specification compliance, schema validity, and link integrity.

**Tech Stack:** Markdown, YAML, Python 3.11, pytest, `pyyaml`, `uv`.

**Spec:** [`docs/superpowers/specs/2026-09-27-repo-agent-skills-design.md`](file:///home/tangc/slm-vlm-post-training/docs/superpowers/specs/2026-09-27-repo-agent-skills-design.md)

## Global Constraints

- Directory base: `.agents/skills/<skill-name>/`
- YAML frontmatter: `name` (1–64 chars, lowercase alphanumeric + hyphens), `description` (starts with `"Use when..."`, third person, concrete triggers, no workflow summary).
- Progressive disclosure: `references/` files must be linked directly (1-level deep) from `SKILL.md`.
- No nested reference chaining.
- Every skill must have a concrete, executable verification gate.
- UTF-8 encoding across all files.

---

### Task 1: Agent Skills Validation Test Harness

**Files:**
- Create: `tests/test_agent_skills.py`

**Interfaces:**
- Consumes: `.agents/skills/` directory structure and YAML frontmatter.
- Produces: Pytest test suite asserting compliance with `agentskills.io` standard and link integrity across all 7 skills.

- [ ] **Step 1: Write the validation test suite**

```python
import os
import re
import pytest
import yaml

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", ".agents", "skills")
EXPECTED_SKILLS = [
    "unsloth-sft",
    "grpo-reasoning-rl",
    "model-export-gguf",
    "character-rp",
    "story-copilot",
    "story-rp-backend",
    "story-rp-frontend",
]

def test_skills_directory_exists():
    assert os.path.isdir(SKILLS_DIR), f"Skills directory not found at {SKILLS_DIR}"

@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_skill_structure_and_frontmatter(skill_name):
    skill_path = os.path.join(SKILLS_DIR, skill_name)
    assert os.path.isdir(skill_path), f"Skill directory missing: {skill_name}"
    
    skill_md_path = os.path.join(skill_path, "SKILL.md")
    assert os.path.isfile(skill_md_path), f"SKILL.md missing in {skill_name}"
    
    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Verify YAML frontmatter delimiters
    assert content.startswith("---\n"), f"SKILL.md in {skill_name} must start with YAML frontmatter"
    parts = content.split("---\n", 2)
    assert len(parts) >= 3, f"SKILL.md in {skill_name} must have closing frontmatter delimiter"
    
    frontmatter = yaml.safe_load(parts[1])
    assert isinstance(frontmatter, dict), f"Frontmatter in {skill_name} is not a valid dict"
    
    # Frontmatter name assertions
    assert "name" in frontmatter, f"'name' missing in {skill_name} frontmatter"
    assert frontmatter["name"] == skill_name, f"Frontmatter name '{frontmatter['name']}' != directory '{skill_name}'"
    assert re.match(r"^[a-z0-9-]+$", frontmatter["name"]), f"Invalid skill name format: {frontmatter['name']}"
    assert len(frontmatter["name"]) <= 64, f"Skill name exceeds 64 chars in {skill_name}"
    
    # Frontmatter description assertions
    assert "description" in frontmatter, f"'description' missing in {skill_name} frontmatter"
    desc = frontmatter["description"]
    assert desc.startswith("Use when"), f"Description in {skill_name} must start with 'Use when...'"
    assert len(desc) <= 1024, f"Description in {skill_name} exceeds 1024 chars"

@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_openai_agent_yaml(skill_name):
    openai_yaml_path = os.path.join(SKILLS_DIR, skill_name, "agents", "openai.yaml")
    assert os.path.isfile(openai_yaml_path), f"agents/openai.yaml missing in {skill_name}"
    
    with open(openai_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    assert "interface" in data, f"'interface' missing in {openai_yaml_path}"
    interface = data["interface"]
    assert "display_name" in interface and interface["display_name"]
    assert "short_description" in interface and interface["short_description"]
    assert "default_prompt" in interface and interface["default_prompt"]

@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_reference_links_resolve(skill_name):
    skill_path = os.path.join(SKILLS_DIR, skill_name)
    skill_md_path = os.path.join(skill_path, "SKILL.md")
    
    if not os.path.isfile(skill_md_path):
        pytest.skip(f"SKILL.md does not exist yet for {skill_name}")
        
    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all reference links pointing to references/
    ref_matches = re.findall(r"`?references/([a-zA-Z0-9_-]+\.md)`?", content)
    assert len(ref_matches) > 0, f"Skill {skill_name} must reference at least one guide in references/"
    
    for ref_file in ref_matches:
        ref_path = os.path.join(skill_path, "references", ref_file)
        assert os.path.isfile(ref_path), f"Referenced file missing: {ref_path} in skill {skill_name}"
```

- [ ] **Step 2: Run test to verify it fails (RED)**

Run: `uv run pytest tests/test_agent_skills.py -v`
Expected: FAIL (Directory `.agents/skills` missing or skills missing)

- [ ] **Step 3: Commit initial test harness**

```bash
git add tests/test_agent_skills.py
git commit -m "test: add validation test harness for agent skills framework"
```

---

### Task 2: Implement `unsloth-sft` Skill

**Files:**
- Create: `.agents/skills/unsloth-sft/SKILL.md`
- Create: `.agents/skills/unsloth-sft/agents/openai.yaml`
- Create: `.agents/skills/unsloth-sft/references/sft-config-schema.md`
- Create: `.agents/skills/unsloth-sft/references/dataset-formatting.md`

**Interfaces:**
- Consumes: `src/slm_post_train/trainers/sft_runner.py`, `configs/sft/smoke_test.yaml`
- Produces: Complete operational guide and reference documentation for Unsloth SFT post-training.

- [ ] **Step 1: Create skill directories**

```bash
mkdir -p .agents/skills/unsloth-sft/agents .agents/skills/unsloth-sft/references
```

- [ ] **Step 2: Create `openai.yaml`**

```yaml
interface:
  display_name: "Unsloth SFT Post-Training"
  short_description: "Supervised fine-tuning with Unsloth, QLoRA, and responses-only loss masking."
  default_prompt: "Use $unsloth-sft to configure, run, or debug supervised fine-tuning with Unsloth."
```

- [ ] **Step 3: Create `references/sft-config-schema.md`**
Define complete YAML config schema, parameter explanations, memory limits (4-bit, paged_adamw_8bit), and LoRA targets.

- [ ] **Step 4: Create `references/dataset-formatting.md`**
Define conversational ShareGPT/HF format, system/user/assistant message structure, and `train_on_responses_only` mechanics.

- [ ] **Step 5: Create `SKILL.md`**
Write SDO-compliant YAML frontmatter (`name: unsloth-sft`, `description: Use when designing, configuring, running, or debugging Supervised Fine-Tuning (SFT) for SLMs or VLMs with Unsloth, including responses-only loss masking, conversational ShareGPT/HF message formatting, 4-bit QLoRA, memory budgeting for <=8GB VRAM, or training YAML configs.`), workflow, guardrails, minimal CLI, and references list.

- [ ] **Step 6: Run verification test**

Run: `uv run pytest tests/test_agent_skills.py -k unsloth_sft -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/unsloth-sft/
git commit -m "feat(skills): add unsloth-sft skill and references"
```

---

### Task 3: Implement `grpo-reasoning-rl` Skill

**Files:**
- Create: `.agents/skills/grpo-reasoning-rl/SKILL.md`
- Create: `.agents/skills/grpo-reasoning-rl/agents/openai.yaml`
- Create: `.agents/skills/grpo-reasoning-rl/references/reward-functions.md`
- Create: `.agents/skills/grpo-reasoning-rl/references/grpo-config-schema.md`

**Interfaces:**
- Consumes: `src/slm_post_train/trainers/grpo_runner.py`, `src/slm_post_train/rewards/`, `configs/grpo/smoke_test.yaml`
- Produces: Complete guide for GRPO online RL, `@register_reward` authoring, and reward monitoring.

- [ ] **Step 1: Create skill directories**

```bash
mkdir -p .agents/skills/grpo-reasoning-rl/agents .agents/skills/grpo-reasoning-rl/references
```

- [ ] **Step 2: Create `openai.yaml`**

```yaml
interface:
  display_name: "GRPO Reasoning RL"
  short_description: "Group Relative Policy Optimization with Unsloth and custom reward verifiers."
  default_prompt: "Use $grpo-reasoning-rl to implement or tune GRPO online reinforcement learning workflows."
```

- [ ] **Step 3: Create `references/reward-functions.md`**
Document `@register_reward`, XML structure checks (`<think>`, `<answer>`), exact match scoring, isolated subprocess code execution, and reward hacking mitigations.

- [ ] **Step 4: Create `references/grpo-config-schema.md`**
Document `num_generations`, `max_completion_length`, `beta`, `temperature`, and `GRPOMonitorCallback`.

- [ ] **Step 5: Create `SKILL.md`**
Write SDO-compliant YAML frontmatter (`name: grpo-reasoning-rl`, `description: Use when designing, implementing, configuring, or debugging Group Relative Policy Optimization (GRPO) reinforcement learning for reasoning SLMs, including prompt-only dataset formatting, custom @register_reward functions, XML structure verification, code execution verifiers, or KL divergence monitoring.`), workflow, guardrails, and reference links.

- [ ] **Step 6: Run verification test**

Run: `uv run pytest tests/test_agent_skills.py -k grpo_reasoning_rl -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/grpo-reasoning-rl/
git commit -m "feat(skills): add grpo-reasoning-rl skill and references"
```

---

### Task 4: Implement `model-export-gguf` Skill

**Files:**
- Create: `.agents/skills/model-export-gguf/SKILL.md`
- Create: `.agents/skills/model-export-gguf/agents/openai.yaml`
- Create: `.agents/skills/model-export-gguf/references/export-workflows.md`
- Create: `.agents/skills/model-export-gguf/references/llama-cpp-serving.md`

**Interfaces:**
- Consumes: `src/slm_post_train/export/exporter.py`, `scripts/infer_gguf.py`, `run_sh/serve_llama_cpp.sh`
- Produces: Guide for adapter merging, GGUF quantization, and local serving.

- [ ] **Step 1: Create skill directories**

```bash
mkdir -p .agents/skills/model-export-gguf/agents .agents/skills/model-export-gguf/references
```

- [ ] **Step 2: Create `openai.yaml`**

```yaml
interface:
  display_name: "Model Export & GGUF Quantization"
  short_description: "Merge LoRA adapters and export quantized GGUF models for llama.cpp and Ollama."
  default_prompt: "Use $model-export-gguf to merge adapters or export models to GGUF format."
```

- [ ] **Step 3: Create `references/export-workflows.md`**
Document `slm-post-train export` CLI options: `--export-type` (`lora`, `merged_16bit`, `merged_4bit`, `gguf`), `--quantization-type` (`q4_k_m`, `q8_0`, `f16`).

- [ ] **Step 4: Create `references/llama-cpp-serving.md`**
Document `run_sh/serve_llama_cpp.sh`, llama-server parameters, Ollama Modelfiles, and testing with `scripts/infer_gguf.py`.

- [ ] **Step 5: Create `SKILL.md`**
Write SDO-compliant YAML frontmatter (`name: model-export-gguf`, `description: Use when exporting, merging, quantizing, or verifying trained SLM/VLM models, including saving 16-bit or 4-bit merged checkpoints, converting to GGUF format (q4_k_m, q8_0, f16), serving via llama.cpp or Ollama, or testing local inference.`), workflow, guardrails, and reference links.

- [ ] **Step 6: Run verification test**

Run: `uv run pytest tests/test_agent_skills.py -k model_export_gguf -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/model-export-gguf/
git commit -m "feat(skills): add model-export-gguf skill and references"
```

---

### Task 5: Implement `character-rp` Skill

**Files:**
- Create: `.agents/skills/character-rp/SKILL.md`
- Create: `.agents/skills/character-rp/agents/openai.yaml`
- Create: `.agents/skills/character-rp/references/character-card-v2-spec.md`
- Create: `.agents/skills/character-rp/references/lorebook-matching-rules.md`

**Interfaces:**
- Consumes: `src/story_rp_engine/rp/character.py`, `src/story_rp_engine/rp/lorebook.py`, `src/story_rp_engine/rp/prompt_builder.py`, `src/story_rp_engine/rp/callbacks.py`
- Produces: Guide for Character Card V2 schemas, macro substitutions, dynamic lorebook keyword matching, and Author's Note steering.

- [ ] **Step 1: Create skill directories**

```bash
mkdir -p .agents/skills/character-rp/agents .agents/skills/character-rp/references
```

- [ ] **Step 2: Create `openai.yaml`**

```yaml
interface:
  display_name: "Character Card & Lorebook RP"
  short_description: "Character Card V2 parsing, dynamic lorebook keywords, and prompt macro injection."
  default_prompt: "Use $character-rp to configure character cards, lorebooks, or roleplay prompt templates."
```

- [ ] **Step 3: Create `references/character-card-v2-spec.md`**
Document Character Card V2 JSON specification, `name`, `description`, `personality`, `scenario`, `first_mes`, `alternate_greetings`, `mes_example`, and macro substitution (`{{char}}`, `{{user}}`).

- [ ] **Step 4: Create `references/lorebook-matching-rules.md`**
Document keyword matching algorithms, whole-word matching, regex keys, selective recursion, token budget caps, and ADK `before_model_callback`.

- [ ] **Step 5: Create `SKILL.md`**
Write SDO-compliant YAML frontmatter (`name: character-rp`, `description: Use when creating, editing, debugging, or testing Character Card V2 JSON definitions, dynamic lorebook keyword matching, {{char}}/{{user}} prompt macro substitution, Author's Note injection, or Google ADK RP agent callbacks.`), workflow, guardrails, and reference links.

- [ ] **Step 6: Run verification test**

Run: `uv run pytest tests/test_agent_skills.py -k character_rp -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/character-rp/
git commit -m "feat(skills): add character-rp skill and references"
```

---

### Task 6: Implement `story-copilot` Skill

**Files:**
- Create: `.agents/skills/story-copilot/SKILL.md`
- Create: `.agents/skills/story-copilot/agents/openai.yaml`
- Create: `.agents/skills/story-copilot/references/director-writer-pipeline.md`
- Create: `.agents/skills/story-copilot/references/story-workflow-graph.md`

**Interfaces:**
- Consumes: `src/story_rp_engine/story/director_agent.py`, `src/story_rp_engine/story/writer_agent.py`, `src/story_rp_engine/story/workflow.py`
- Produces: Guide for Google ADK 2.0 Director/Writer multi-agent narrative workflow graph.

- [ ] **Step 1: Create skill directories**

```bash
mkdir -p .agents/skills/story-copilot/agents .agents/skills/story-copilot/references
```

- [ ] **Step 2: Create `openai.yaml`**

```yaml
interface:
  display_name: "Story Co-Pilot Workflow"
  short_description: "Google ADK 2.0 Director and Writer collaborative storytelling pipeline."
  default_prompt: "Use $story-copilot to build, configure, or tune the Director/Writer story generation workflow."
```

- [ ] **Step 3: Create `references/director-writer-pipeline.md`**
Document Director prompt engineering, pacing/tone guidance output format, and Writer prose generation prompt constraints.

- [ ] **Step 4: Create `references/story-workflow-graph.md`**
Document ADK graph structure, node execution order, state dictionary transitions, branching scene logic, and human review loop.

- [ ] **Step 5: Create `SKILL.md`**
Write SDO-compliant YAML frontmatter (`name: story-copilot`, `description: Use when designing, configuring, or modifying the Google ADK 2.0 Story Co-Pilot workflow, including Story Director planning, Story Writer prose generation, narrative workflow graphs, scene pacing guidance, or story branching states.`), workflow, guardrails, and reference links.

- [ ] **Step 6: Run verification test**

Run: `uv run pytest tests/test_agent_skills.py -k story_copilot -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/story-copilot/
git commit -m "feat(skills): add story-copilot skill and references"
```

---

### Task 7: Implement `story-rp-backend` Skill

**Files:**
- Create: `.agents/skills/story-rp-backend/SKILL.md`
- Create: `.agents/skills/story-rp-backend/agents/openai.yaml`
- Create: `.agents/skills/story-rp-backend/references/api-endpoints-and-sse.md`
- Create: `.agents/skills/story-rp-backend/references/database-session-storage.md`

**Interfaces:**
- Consumes: `src/story_rp_engine/api/`, `src/story_rp_engine/storage/store.py`, `src/story_rp_engine/core/model_provider.py`, `run_sh/run_story_rp_backend.sh`
- Produces: Guide for FastAPI REST routes, SSE streaming, SQLite persistence, and LiteLLM model provider routing.

- [ ] **Step 1: Create skill directories**

```bash
mkdir -p .agents/skills/story-rp-backend/agents .agents/skills/story-rp-backend/references
```

- [ ] **Step 2: Create `openai.yaml`**

```yaml
interface:
  display_name: "Story & RP Backend Engine"
  short_description: "FastAPI REST API, SSE streaming, SQLite persistence, and LiteLLM model routing."
  default_prompt: "Use $story-rp-backend to develop, test, or troubleshoot the FastAPI backend service."
```

- [ ] **Step 3: Create `references/api-endpoints-and-sse.md`**
Document REST route specifications (`/api/rp/chat/stream`, `/api/story/stream`, `/api/lorebook`), request/response schemas, and `sse-starlette` chunk formatting (`event: message`, `data: {"chunk": "..."}`, `event: done`).

- [ ] **Step 4: Create `references/database-session-storage.md`**
Document `DatabaseSessionService`, SQLite table schema (`sessions`, `messages`, `snapshots`), async CRUD operations via `aiosqlite`, and LiteLLM failover config.

- [ ] **Step 5: Create `SKILL.md`**
Write SDO-compliant YAML frontmatter (`name: story-rp-backend`, `description: Use when developing, debugging, or extending the FastAPI backend for the Story and RP engine, including REST routes, Server-Sent Events (SSE) streaming, SQLite session storage via DatabaseSessionService, LiteLLM model provider routing, or Phoenix OpenTelemetry tracing.`), workflow, guardrails, and reference links.

- [ ] **Step 6: Run verification test**

Run: `uv run pytest tests/test_agent_skills.py -k story_rp_backend -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/story-rp-backend/
git commit -m "feat(skills): add story-rp-backend skill and references"
```

---

### Task 8: Implement `story-rp-frontend` Skill

**Files:**
- Create: `.agents/skills/story-rp-frontend/SKILL.md`
- Create: `.agents/skills/story-rp-frontend/agents/openai.yaml`
- Create: `.agents/skills/story-rp-frontend/references/ui-components-and-modes.md`
- Create: `.agents/skills/story-rp-frontend/references/sse-event-client.md`

**Interfaces:**
- Consumes: `src/story_rp_engine/web/index.html`, `src/story_rp_engine/web/app.js`, `src/story_rp_engine/web/style.css`
- Produces: Guide for web UI architecture, dual-mode switching (RP Chat vs Story Co-Pilot), and client-side SSE streaming reader.

- [ ] **Step 1: Create skill directories**

```bash
mkdir -p .agents/skills/story-rp-frontend/agents .agents/skills/story-rp-frontend/references
```

- [ ] **Step 2: Create `openai.yaml`**

```yaml
interface:
  display_name: "Story & RP Web Frontend"
  short_description: "Single-page dual-mode web interface, SSE streaming reader, and card manager."
  default_prompt: "Use $story-rp-frontend to modify UI components, styles, or client-side event streaming."
```

- [ ] **Step 3: Create `references/ui-components-and-modes.md`**
Document HTML layout hierarchy, dual-mode tabs (RP chat vs Story Co-Pilot), character card importer modal, lorebook editor, and CSS custom properties theme variables.

- [ ] **Step 4: Create `references/sse-event-client.md`**
Document client-side SSE reader via `fetch` ReadableStream / `TextDecoder`, chunk buffering, real-time Markdown rendering, and disconnect/error handling.

- [ ] **Step 5: Create `SKILL.md`**
Write SDO-compliant YAML frontmatter (`name: story-rp-frontend`, `description: Use when creating, modifying, or styling the browser interface for the Story and RP engine in src/story_rp_engine/web/, including dual-mode UI switching, client-side SSE streaming readers, markdown rendering, character card import modals, or CSS layout improvements.`), workflow, guardrails, and reference links.

- [ ] **Step 6: Run verification test**

Run: `uv run pytest tests/test_agent_skills.py -k story_rp_frontend -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/story-rp-frontend/
git commit -m "feat(skills): add story-rp-frontend skill and references"
```

---

### Task 9: Full Suite Verification & Documentation

**Files:**
- Modify: `README.md` (Add Agent Skills section linking to `.agents/skills/`)

- [ ] **Step 1: Run complete agent skills pytest suite**

Run: `uv run pytest tests/test_agent_skills.py -v`
Expected: All tests PASS (7 skills, frontmatter valid, references resolve, openai.yaml valid)

- [ ] **Step 2: Run existing project tests to guarantee zero regressions**

Run: `uv run pytest tests/story_rp_engine/ -v`
Expected: All existing tests PASS

- [ ] **Step 3: Update README.md with Agent Skills documentation**
Document the presence of `.agents/skills/`, how agents discover and load them, and list the 7 available skills.

- [ ] **Step 4: Commit and finalize**

```bash
git add README.md tests/test_agent_skills.py
git commit -m "docs: document agent skills framework in README"
```
