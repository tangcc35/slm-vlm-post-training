# RP Group Chat Design

**Date:** 2026-10-06
**Status:** Approved design, pending implementation plan
**Scope:** `src/story_rp_engine/`, `tests/story_rp_engine/`, `CLAUDE.md`, `.agents/skills/character-rp/`

## Goal

The user roleplays with several characters in one scene. Each turn, an LLM speaker selector picks which members reply and in what order. Each chosen character then writes its own reply, in character, after seeing what the user and the earlier speakers said.

Success: a user saves a group of 2+ characters, opens a group chat, sees the right characters answer in a sensible order with each reply labelled by speaker, and can reopen, rename, delete, or trim the chat from history.

Out of scope: per-group selector instructions, sample groups in `examples/`, a non-streaming chat endpoint.

## Decisions

- **Groups are saved entities** (`GroupCard`), edited in their own tab, not picked ad hoc per chat.
- **The speaker selector is an `LlmAgent` with structured output**, a list of member `char_id`s.
- **One ADK dynamic workflow per group**, cached in `AgentRegistry` by `group_id` and rebuilt when the stored group or a member card changes. This is the same staleness check RP agents use, kept for consistency with RP.
- **Separate ADK app `group_app`.** Existing `rp_app` chats are untouched.

## 1. Data model, storage, registry

### Types (`core/types.py`)

```python
class GroupCard(BaseModel):
    group_id: str
    name: str
    char_ids: List[str]            # fallback speaking order
    scenario: str = ""             # shared scene; replaces each card's own scenario
    first_mes: str = ""            # group opening message
    lorebook_id: Optional[str] = None  # preselected for new chats

class GroupChatRequest(BaseModel):
    group_id: str
    session_id: str
    message: str
    authors_note: Optional[str] = None
    lorebook_id: Optional[str] = None  # same semantics as RPChatRequest.lorebook_id
    user_name: Optional[str] = "User"
    greeting: Optional[str] = None
    chunk_size: Optional[int] = 16
```

### Storage (`storage/store.py`)

`EngineStore` gets a third collection next to characters and lorebooks: a `story_rp_groups` table when a DB URL is set, otherwise JSON files under `<STORY_RP_STORAGE_DIR>/groups/`. It has `save_group`, `get_group`, `list_groups` and `delete_group`, built on the existing `_FileCollection` / `_SqlCollection`.

### Registry (`core/agent_registry.py`)

- `get_or_create_group_runner(group_id) -> Runner` loads the group and its member cards from the store on every call. The cached workflow and runner are kept with the `(group, cards)` snapshot they were built from, and both are rebuilt when the snapshot differs.
- If a member's card no longer exists, that member is skipped. If the group is missing, or no members remain, it raises `ValueError("Group <id> not found")`, which the route maps to 404.
- `forget_group(group_id)` drops the cached workflow and runner. The group delete route calls it.
- The runner uses `App(name="group_app", root_agent=workflow, events_compaction_config=...)`. Compaction is turn-count only, like `story_app`, because token-threshold compaction could run between two characters' model calls. The summarizer uses the new `GROUP_SUMMARY_PROMPT`. It asks for a summary of the scene for the characters (who is present, events in order, relationships, unresolved threads, user preferences) and tells the summarizer to ignore `speaker_selector` output.

## 2. Workflow, selector, character prompts

New package `src/story_rp_engine/group/`.

### Workflow (`group/workflow.py`)

```python
def create_group_workflow(group, cards, config) -> Workflow:
    selector = create_speaker_selector(group, cards, config)
    agents = {c.char_id: create_group_char_agent(c, group, cards, config) for c in cards}

    @node(rerun_on_resume=True)
    async def group_turn(ctx: Context):
        # Sub-branch: the selector reads the main conversation, but characters never see its JSON.
        plan = await ctx.run_node(selector, use_sub_branch=True)
        for char_id in pick_speakers(plan, [c.char_id for c in cards]):
            await ctx.run_node(agents[char_id])

    return Workflow(name="group_workflow", edges=[("START", group_turn)])
```

- Characters run one after another on the main branch, so each one sees the user's message and the replies before it. They must not run in parallel: `asyncio.gather` runs would get sub-branches and hide the replies from each other.
- With `use_sub_branch=True`, the selector's events go on a sub-branch. `_is_event_belongs_to_branch` shows main-branch events to the selector but hides the selector's events from main-branch characters. Each turn's selector branch is a sibling of the others, so the selector doesn't see its own earlier picks either.
- `pick_speakers(plan, member_ids)` keeps the IDs that are members, drops duplicates, keeps the order, and returns `member_ids` when the result is empty. If the selector returns JSON that doesn't parse, ADK raises and the error is reported in the stream like any model error. There's no retry.

### Agent names

A character agent is named `char_<char_id with [^a-zA-Z0-9_] replaced by _>`. Agent names must be identifiers. `char_id`s are stable while display names may be non-ASCII, and deriving names from `char_id` rather than member position keeps saved event authors correct after the group is edited. `group_agent_name(char_id)` is the single helper that computes this; the routes use it to map event authors back to `char_id`. Two IDs that sanitize to the same name are a known limitation that isn't handled.

### Speaker selector (`group/selector_agent.py`)

- `LlmAgent(name="speaker_selector", include_contents="default", output_schema=SpeakerPlan)`, using `get_adk_model` and `get_generate_config`.
- `SpeakerPlan` is built per group, with `speakers: List[Literal[<member char_ids>]]`, so backends with constrained decoding (Gemini, llama.cpp) can only emit valid IDs.
- Instruction: the members (ID, name, description cut to a few hundred characters), the group scenario, and the user's name, then the task. Choose the characters who would naturally respond to the latest message, in speaking order. Whoever the user addresses goes first. Leave out characters with nothing to add. Return at least one.

### Character agents (`group/char_agent.py`)

- `LlmAgent(name=group_agent_name(card.char_id), include_contents="default", before_model_callback=rp_before_model_callback)`, with a callable instruction (as in RP, so braces in card text are safe).
- `build_group_system_instruction(card, group, cards, user_name, greeting)` in `group/prompt_builder.py`, reusing `rp.prompt_builder.replace_macros`. Compared with `build_rp_system_instruction`:
  - **Intro:** "You are {char} in a group roleplay with {user} and {other names}. Write only {char}'s dialogue, actions and thoughts. Never write the words, actions or decisions of {user} or the other characters." It also says that lines marked `[char_x] said:` come from the other characters, and lists each agent name with its display name.
  - **Scenario:** the group's `scenario` replaces the card's.
  - **Opening message:** the group's `first_mes`, through the `greeting` state key as in RP, is presented as the scene's shared opening message, not one this character sent.
  - **Unchanged:** the card's system prompt, description, personality, example dialogue, guidelines and post-history instructions.
- Lorebook and author's note: `rp_before_model_callback` is reused unchanged. For the second and later speakers, the latest user-role message is the previous character's line (ADK presents other agents as user role), so lore is matched against that line and the note is attached to it. This is accepted: lore the previous speaker raised is relevant, and the note still sits at the end of the context.

## 3. API, streaming, sessions

New `api/routes_group.py`, registered in `api/app.py`.

### Group CRUD

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/groups` | `group_id` required (400 if missing), like characters |
| GET | `/api/v1/groups` | dict of `group_id → GroupCard` |
| GET | `/api/v1/groups/{group_id}` | 404 if missing |
| DELETE | `/api/v1/groups/{group_id}` | also `registry.forget_group` |

### Chat

`POST /api/v1/group/chat/stream` with a `GroupChatRequest` body.

- `_group_state_delta` mirrors `_rp_state_delta`. It sets `authors_note`, `user_name`, `greeting`, `group_id` and `last_message`, plus the lorebook copy and `lorebook_id` when `lorebook_id` is sent.
- `stream_group_turn(runner, user_id, session_id, message, state_delta, speakers)` in `core/agent_utils.py`. `speakers` maps agent name → `char_id`. It yields `(char_id, text)` pairs with the same partial/final handling as `stream_runner_turn`, and skips events from authors not in the map (the selector).
- `format_sse_stream` also accepts `(speaker, text)` pairs. When it gets them, each delta carries `speaker`, the buffer is flushed whenever the speaker changes, and the final event lists the replies:

  ```
  data: {"speaker": "alice", "delta": "Hi there"}
  data: {"speaker": "bob", "delta": "Hey."}
  data: {"replies": [{"speaker": "alice", "text": "Hi there"}, {"speaker": "bob", "text": "Hey."}], "done": true}
  data: [DONE]
  ```

  Plain string chunks (RP, story) produce exactly the current output.

### Sessions (`group_app`, `user_id="User"`)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/group/sessions?group_id=` | same fields as the RP list, filtered on `group_id` |
| GET | `/api/v1/group/sessions/{id}/turns` | each turn has `speaker` (`char_id`, or `null` for user), and selector events are excluded |
| PATCH | `/api/v1/group/sessions/{id}` | rename |
| DELETE | `/api/v1/group/sessions/{id}` | delete |
| POST | `/api/v1/group/sessions/{id}/turns/delete` | same semantics as RP |

The RP session route bodies, which hardcode `rp_app`, move into helpers that take `app_name`. `_message_event_indexes` also takes a set of authors to skip. `routes_rp` and `routes_group` both call these helpers, and RP URLs and responses don't change.

## 4. UI (`web/index.html`, `web/app.js`)

Two new top-level tabs. The Roleplay tab is not changed.

- **Groups** (editor, like Lorebooks): a list with search, and a form with `group_id`, name, members (multi-select of saved characters; selection order is the fallback order), scenario, opening message and default lorebook. Save and delete.
- **Group Chat** (layout copied from Roleplay):
  - **Sidebar:** group picker, history list with rename and delete, user name, lorebook (preselected from the group's default) and author's note.
  - **Chat:** the group's `first_mes` is shown as the opening message. Each character reply is its own bubble, labelled with the character's name (looked up from `char_id`). During streaming, a new bubble opens when `speaker` changes. Reopening a chat rebuilds the bubbles from `/turns`, and deleting a message uses `turns/delete`.
  - **Code:** new state and functions use a `group` prefix in `app.js`. The group SSE parser is its own function, so the RP path is untouched.

## 5. Testing and docs

Tests use fake ADK `BaseLlm`s and `InMemorySessionService`, as the existing engine tests do.

- **Workflow:** the selector returns `{"speakers": ["b", "a"]}`, and B then A reply in that order. B's `LlmRequest` contains A's line and no selector JSON. `pick_speakers` drops unknown and duplicate IDs and falls back to group order on an empty list.
- **Prompt:** `build_group_system_instruction` uses the group scenario, not the card's, names the other members and includes the group opening message.
- **Registry:** the runner is cached across calls and rebuilt when a member card changes. A missing member is skipped, and a group with no remaining members raises.
- **Store:** group CRUD round-trips in the file backend, and in the Postgres test when `STORY_RP_TEST_PG_URL` is set.
- **Routes:** the group stream emits `speaker` deltas and a final `replies` event, and `/turns` returns speakers without selector events. The existing RP session and stream tests keep passing.

Docs:
- `CLAUDE.md`, under `story_rp_engine`: a "Group chat" bullet covering the workflow, the selector sub-branch, `group_app` and the agent naming.
- `.agents/skills/character-rp/`: a short section on groups. `tests/test_agent_skills.py` must still pass.
