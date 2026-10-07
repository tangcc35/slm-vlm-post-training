# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Writing code: keep it simple

- Write the simplest code that solves the actual problem: fewer lines, fewer branches, fewer abstractions.
- Don't design for hypothetical use cases or "what if later…" scenarios. Build what's needed now.
- Don't chase rare edge cases. If a simple design has a known limitation, accept it and mention it in one line instead of adding code to cover it.
- Don't be over-protective: no defensive checks for states that can't happen, no re-validating values our own code produced, no try/except around code that isn't expected to fail. Validate only at real boundaries (request bodies, external APIs, user-supplied files).
- Keep changes scoped to the task. No drive-by refactors, new config options, helpers, or abstractions unless asked.
- Tests cover the behavior that changed, not every permutation.


## Answering questions

- Answer concisely: lead with the direct answer, then only the reasoning needed to support it.
- Keep the logic tight: one clear line of reasoning, no restating, no tangents or options I didn't ask about.


## Commands

Python 3.11, managed with `uv`. No linter or formatter is configured.

```bash
uv sync --extra dev                                    # install (add --extra phoenix for a local Phoenix server)
uv run pytest                                          # full suite, ~20s
uv run pytest tests/story_rp_engine/test_lorebook.py   # one file
uv run pytest tests/test_rewards.py::test_name         # one test

uv run slm-post-train train --config configs/sft/smoke_test.yaml    # 2-step real SFT run (needs CUDA GPU)
uv run slm-post-train train --config configs/grpo/smoke_test.yaml   # 2-step real GRPO run
uv run slm-post-train curate-data --config configs/sft/qwen35_08b_nsfw_story.yaml
uv run slm-post-train export --model-path <dir> --output-dir <dir> --format gguf --quant q4_k_m

./run_sh/run_story_rp_backend.sh   # engine on :8000 with --reload; sources .env, starts Phoenix if PHOENIX_ENABLED
./run_sh/serve_llama_cpp.sh        # llama.cpp server (LLAMA_* vars in .env) plus the engine
```

Testing notes:
- Tests never train or call a real LLM: trainer tests patch Unsloth/TRL, engine tests use fake ADK `BaseLlm`s and `InMemorySessionService`. Async tests use `@pytest.mark.anyio`.
- Postgres storage tests are skipped unless `STORY_RP_TEST_PG_URL` points at a Postgres server.
- `EngineConfig` skips `load_dotenv()` under pytest, so `.env` doesn't leak into tests.

## Architecture

Two packages under `src/` ship in one wheel but share no code. The engine can serve models trained by the first package once they're exported to GGUF and served through llama.cpp or another OpenAI-compatible server.

### `slm_post_train`

- `cli.py` dispatches: `train` reads the YAML's `stage` and calls `trainers/sft_runner.run_sft` or `trainers/grpo_runner.run_grpo`; `curate-data` calls `data/nsfw_story.curate_from_config`; `export` calls `export/exporter.export_model`.
- Runners take the raw YAML dict and read each section (`model`, `lora`, `dataset`, `training`, `output`, `rewards`) with `.get(key, default)`. There's no schema: defaults live in the runner code, and a new config knob means reading it in the runner.
- `import unsloth` must come before any `trl`/`transformers` import so its kernel patches apply. `slm_post_train/__init__.py` does this and also patches `trl.import_utils` for transformers ≥ 5.
- `models/loader.py` loads `FastLanguageModel` or `FastVisionModel` (chosen by `model.modality`) and attaches LoRA when the `lora` section is non-empty.
- SFT data (`data/sft_data.py`) converts `messages`, `instruction`/`output`, `prompt`/`response` and ShareGPT rows into a `conversations` column, then renders a `text` column with the Unsloth chat template. Responses-only loss takes its delimiters from the tokenizer's Unsloth parts, then from `CHAT_TEMPLATE_DELIMITERS[dataset.chat_template]`, then from explicit `instruction_part`/`response_part`. It's turned off when `packing: true`.
- GRPO data (`data/grpo_data.py`) maps `prompt`/`question`/`problem` to a chat-format `prompt` and `answer`/`solution`/`ground_truth` to `answer`. Rewards are resolved by name from `rewards/registry.py`. Built-ins register because `rewards/__init__.py` imports `standard.py`, so a new reward module must also be imported before the runner looks it up.
- Outputs: SFT and export each write into a timestamped `YYYYMMDD-HHMMSS` subdirectory of the configured dir; GRPO doesn't. SFT logs to W&B (project `slm_post_train`); GRPO doesn't log anywhere. GGUF export uses llama.cpp from `~/.unsloth/llama.cpp`.

### `story_rp_engine`

- `api/app.py:create_app(store=None, config=None)` is a factory (uvicorn runs it with `--factory`), so tests can inject a store and config. It puts `store`, `config`, `session_service`, `agent_registry` and the story `runner` on `app.state`, and serves `web/` (a vanilla JS UI) at `/`.
- `core/config.py:EngineConfig` reads every env var in a `default_factory`, so env values are captured when it's instantiated.
- `core/model_provider.get_adk_model`: model names starting with `gemini`, `gemini/` or `google/` use ADK's native `Gemini`; everything else goes through ADK's `LiteLlm`. Every agent uses the same sampling settings from `get_generate_config`.
- `core/agent_registry.AgentRegistry` caches one RP `LlmAgent` and `Runner` per `char_id` (ADK app `rp_app`), plus one story workflow runner (`story_app`) and one group chat runner per `group_id` (`group_app`). An RP agent or group runner loaded from the store is rebuilt when the stored card (or group, or a member's card) changes, because serverless instances can share the store. `rp_app` uses `EventsCompactionConfig` with a token threshold; the workflow apps use turn-count compaction only (see below).
- Roleplay (`rp/`): the agent's instruction is a **callable** (`build_rp_system_instruction`), so ADK's `{var}` state templating can't choke on braces in card text. Per-turn context (lorebook entries whose keys match, and the author's note) is appended to the latest user message by `rp_before_model_callback`, not to the system prompt. That keeps the system prompt identical across turns, so prompt caching works. Session state keys (`user_name`, `user_persona`, `persona_id`, `greeting`, `authors_note`, `lorebook`) are set through `state_delta` in `api/chat_sessions.chat_state_delta`, which resolves the request's `persona_id` into the saved persona's name and description every turn; the description goes in the system prompt as `<user_persona>`.
- Story (`story/`): a dynamic ADK `Workflow` whose one step, `story_turn`, calls `ctx.run_node(director)` with no input and then `ctx.run_node(writer, notes)`. The director (`include_contents="default"`) reads the whole session history, compacted; the writer sees only the director's notes. Both use ADK `{key?}` templating over the state keys `premise`, `genre`, `tone`, `current_text` and `instruction`. The director's `director_before_model_callback` appends lorebook entries whose keys appear in the user's message to that message (reusing the helpers in `rp/callbacks`); the director restates what the writer needs in its notes, and the writer never sees the lorebook. The UI is a chat: the first turn carries the setup (`premise`, `genre`, `tone`, `lorebook_ids`), which stays in session state, and later turns send only `instruction`. `routes_story` builds `current_text` from the session's earlier `story_writer` replies (only the last `RECENT_TEXT_CHARS`), and passes `author="story_writer"` so `core/agent_utils` surfaces only the writer's events. `story_app` uses turn-count compaction only, with `STORY_SUMMARY_PROMPT`, because token-threshold compaction can run between the director and the writer and summarize away the notes.
- Group chat (`group/`): a saved `GroupCard` (`group_id`, `name`, `char_ids`, `scenario`, `lorebook_ids`) puts several characters in one scene; it has no greeting, so the user's first message opens the scene. `create_group_workflow` builds a dynamic `Workflow` per group whose `group_turn` step runs the `speaker_selector` `LlmAgent` (its `output_schema` is `{"speakers": List[Literal[<member ids>]]}`), then `ctx.run_node`s each chosen character in order; `pick_speakers` drops repeats and falls back to group order. Characters run on the branch `group` and the selector on a sub-branch of it, so the selector sees every line but characters never see its JSON (a node with no branch would see everything). Character agents are named `group_agent_name(char_id)` (`char_` + the ID with non-word characters replaced, so CJK IDs stay distinct), use `build_group_system_instruction` (the group's scenario replaces the card's) and reuse `rp_before_model_callback`. `group_app` uses turn-count compaction with `group_summary_prompt(cards)`.
- Turns run through `execute_runner_turn` / `stream_runner_turn`. `format_sse_stream` emits `data: {"delta": …}` chunks, then `{"full_text": …, "done": true}`, then `data: [DONE]`; `web/app.js` parses this exact format. Group chat streams `(char_id, text)` pairs from `stream_group_turn`; `format_sse_stream` then adds `"speaker"` to each delta and ends with `{"replies": [{"speaker", "text"}], "done": true}` instead of `full_text`.
- Every session uses `user_id="User"`. The RP and group session routes share the helpers in `api/chat_sessions.py`, each passing its app name (`rp_app`, `group_app`). Deleting a turn rewrites the whole session: delete it, recreate it, and re-append the kept events.
- Storage (`storage/store.py:EngineStore`): when a DB URL is set (`DATABASE_URL`, `STORY_RP_DB_URL` or `POSTGRES_URL`), ADK `DatabaseSessionService` and the `story_rp_characters`/`story_rp_lorebooks`/`story_rp_groups`/`story_rp_personas` tables share one engine. With no DB URL, sessions go to SQLite `sessions.db` and characters, lorebooks, groups and personas to JSON files under `STORY_RP_STORAGE_DIR` (default `.engine_data/`). `storage/db.normalize_db_url` rewrites Neon/libpq URLs for asyncpg. IDs go through `_sanitize_key`.
- On startup the app warms up with one story turn, unless the model is remote or `STORY_RP_SKIP_WARMUP=1`.

### Deployment and other directories

- `vercel/`: the build step copies `src/story_rp_engine` into `vercel/src/`, and `vercel/api/index.py` sets serverless defaults (`/tmp` storage, no warmup, NullPool, Phoenix off). `vercel/requirements.txt` is maintained separately from `pyproject.toml`, so new engine runtime deps must go in both. It leaves out `litellm` on purpose: Vercel only runs Gemini models.
- `.agents/skills/`: [agentskills.io](https://agentskills.io) skills covering each subsystem (SFT, GRPO, GGUF export, character RP, story co-pilot, backend, frontend). `tests/test_agent_skills.py` checks their format and links.
- `examples/story_rp/`: sample cards, lorebooks and scripted demos, loaded and played by `scripts/story_rp_examples.py`. `tests/story_rp_engine/test_examples.py` checks that file names match IDs and that the demos trigger every enabled lorebook entry.
- `docs/superpowers/{specs,plans}/`: dated design specs and implementation plans for past features.
- `unsloth_notebook_reference/`: the Unsloth notebooks (Gemma SFT, GRPO Sudoku) the recipes are based on.
