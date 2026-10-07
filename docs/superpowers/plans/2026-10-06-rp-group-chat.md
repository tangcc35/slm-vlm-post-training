# RP Group Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user roleplay with a saved group of characters. Each turn, an LLM speaker selector picks who replies, then each chosen character streams its own reply.

**Architecture:** A new `story_rp_engine/group/` package builds one ADK dynamic `Workflow` per group. Its `group_turn` node runs the `speaker_selector` agent (structured output) on a sub-branch, then runs each chosen character agent on the shared branch `group`. `AgentRegistry` caches the runner per `group_id` (app `group_app`). A new `routes_group.py` adds group CRUD, a streaming chat endpoint whose SSE deltas carry a `speaker`, and session routes. The RP session code moves into shared helpers that both routers call. Two new UI tabs: Groups (editor) and Group Chat.

**Tech Stack:** Python 3.11, uv, Google ADK 2.7.0 (`Workflow`, `ctx.run_node`, `LlmAgent.output_schema`), FastAPI, Pydantic v2, SQLAlchemy async, pytest + anyio, Vue 3 (CDN) and Tailwind.

**Spec:** `docs/superpowers/specs/2026-10-06-rp-group-chat-design.md`

## Global Constraints

- ADK app name `group_app`. Every session uses `user_id="User"`.
- Character agent name: `"char_" + re.sub(r"\W", "_", char_id)`. Selector agent name: `speaker_selector`.
- Characters run with `ctx.run_node(agent, override_branch="group")`. The selector runs with `ctx.run_node(selector, use_sub_branch=True, override_branch="group")`.
- Selector `output_schema`: `{"speakers": List[Literal[<member char_ids>]]}`.
- Group SSE format: `data: {"speaker": <char_id>, "delta": <text>}` per chunk; final `data: {"replies": [{"speaker", "text"}, ...], "done": true}`; then `data: [DONE]`.
- RP and story SSE output must stay byte-for-byte unchanged.
- Turn-count compaction only for `group_app` (no token threshold).
- No new dependencies. New runtime code lives in `src/story_rp_engine/`, which Vercel copies whole.
- Tests never call a real LLM: they patch `get_adk_model` with fake `BaseLlm`s. Async tests use `@pytest.mark.anyio`.
- Follow `CLAUDE.md`: simplest code, no defensive checks beyond real boundaries, match surrounding style.

## Review Focus

- **Unusable selector reply.** The selector returns text that isn't JSON, or names a non-member. The stream should report an `error` event and end with `[DONE]`, not hang or return 500. Pinned in Task 7 (`test_group_chat_reports_unusable_selector_reply`).
- **Deleted member.** A member's character is deleted after the group was saved. The chat should continue with the remaining members. Pinned in Task 4 (`test_group_runner_skips_deleted_members`).
- **Non-ASCII IDs.** With `char_id`s like `思琪`, agent names should stay valid and distinct, so replies are labelled with the right speaker. Pinned in Task 2 (`test_group_agent_name_keeps_ids_distinct`).
- **Card edited between turns.** A member's card is edited between turns. The next turn should use the new card. Pinned in Task 4 (`test_group_runner_cached_and_rebuilt_when_a_member_changes`).
- **Deleting a message.** The selector's hidden events must not shift the indexes, so deleting a message removes the one the user clicked. Pinned in Task 7 (`test_group_session_rename_delete_turn_and_delete`).

## File Structure

| File | Responsibility |
|---|---|
| `src/story_rp_engine/core/types.py` (modify) | `GroupCard`, `GroupChatRequest` |
| `src/story_rp_engine/storage/store.py` (modify) | `story_rp_groups` table / `groups/` dir; `save_group`, `get_group`, `list_groups`, `delete_group` |
| `src/story_rp_engine/group/__init__.py` (create, empty) | package marker |
| `src/story_rp_engine/group/prompt_builder.py` (create) | `group_agent_name`, `build_group_system_instruction`, `build_selector_instruction`, `group_summary_prompt` |
| `src/story_rp_engine/group/selector_agent.py` (create) | `SPEAKER_SELECTOR`, `create_speaker_selector` |
| `src/story_rp_engine/group/char_agent.py` (create) | `create_group_char_agent` |
| `src/story_rp_engine/group/workflow.py` (create) | `SCENE_BRANCH`, `pick_speakers`, `group_speakers`, `create_group_workflow` |
| `src/story_rp_engine/core/agent_registry.py` (modify) | `get_or_create_group_runner`, `forget_group`, `_turn_count_compaction_config` |
| `src/story_rp_engine/core/agent_utils.py` (modify) | `stream_group_turn`; `format_sse_stream` accepts `(speaker, text)` pairs |
| `src/story_rp_engine/api/chat_sessions.py` (create) | session helpers shared by the RP and group routers |
| `src/story_rp_engine/api/routes_rp.py` (modify) | uses `chat_sessions` helpers, behavior unchanged |
| `src/story_rp_engine/api/routes_group.py` (create) | group CRUD, `/group/chat/stream`, `/group/sessions*` |
| `src/story_rp_engine/api/app.py` (modify) | registers the group router |
| `src/story_rp_engine/web/app.js`, `web/index.html` (modify) | Groups tab, Group Chat tab |
| `tests/story_rp_engine/conftest.py` (create) | `group_models` fixture (fake selector and character LLMs) |
| `tests/story_rp_engine/test_group_prompt_builder.py`, `test_group_workflow.py`, `test_group_api.py` (create) | new tests |
| `tests/story_rp_engine/test_storage.py`, `test_agent_registry.py`, `test_agent_utils.py`, `tests/test_frontend_integration.py` (modify) | new tests |
| `CLAUDE.md`, `.agents/skills/character-rp/SKILL.md` (modify) | docs |

Before Task 1, confirm the baseline is green: `uv run pytest -q`. Expected: all pass, Postgres tests skipped.

---

### Task 1: Group types and storage

**Files:**
- Modify: `src/story_rp_engine/core/types.py` (append after `RPChatRequest`)
- Modify: `src/story_rp_engine/storage/store.py`
- Test: `tests/story_rp_engine/test_storage.py`

**Interfaces:**
- Produces: `GroupCard(group_id: str, name: str, char_ids: List[str] = [], scenario: str = "", first_mes: str = "", lorebook_id: Optional[str] = None)`. `GroupChatRequest(group_id, session_id, message, authors_note=None, lorebook_id=None, user_name="User", greeting=None, chunk_size=16)`. `EngineStore.save_group(group_id, group) -> None`, `get_group(group_id) -> Optional[GroupCard]`, `list_groups() -> Dict[str, GroupCard]`, `delete_group(group_id) -> bool`.

- [ ] **Step 1: Write the failing test.** Append to `tests/story_rp_engine/test_storage.py`, and add `GroupCard` to its `from story_rp_engine.core.types import (...)` list:

```python
@pytest.mark.anyio
async def test_group_crud(make_store):
    store = make_store()
    group = GroupCard(group_id="tavern", name="Tavern", char_ids=["valerie", "思琪"], scenario="A rainy night.")
    await store.save_group("tavern", group)

    assert await store.get_group("tavern") == group
    assert await store.list_groups() == {"tavern": group}

    assert await store.delete_group("tavern") is True
    assert await store.get_group("tavern") is None
    assert await store.delete_group("tavern") is False
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `uv run pytest tests/story_rp_engine/test_storage.py -k group -v`
Expected: collection error `ImportError: cannot import name 'GroupCard'`.

- [ ] **Step 3: Add the types.** Append to `src/story_rp_engine/core/types.py`:

```python
class GroupCard(BaseModel):
    group_id: str
    name: str
    char_ids: List[str] = Field(default_factory=list, description="Members, in the order they speak when the speaker selector picks no one.")
    scenario: str = Field(default="", description="The shared scene; replaces each member card's own scenario.")
    first_mes: str = Field(default="", description="Opening message shown before the user's first message.")
    lorebook_id: Optional[str] = Field(default=None, description="Lorebook selected by default when a new chat with this group starts.")


class GroupChatRequest(BaseModel):
    group_id: str
    session_id: str
    message: str
    authors_note: Optional[str] = None
    lorebook_id: Optional[str] = Field(default=None, description="Loads this lorebook into the session; omit to keep the current one, empty string to clear it.")
    user_name: Optional[str] = "User"
    greeting: Optional[str] = Field(default=None, description="The group's opening message, shown to the user before their first message.")
    chunk_size: Optional[int] = Field(default=16, ge=1, le=100, description="Number of tokens to buffer before yielding in streaming mode.")
```

- [ ] **Step 4: Add the storage.** In `src/story_rp_engine/storage/store.py`:

Change the types import to `from story_rp_engine.core.types import CharacterCard, GroupCard, Lorebook`.

After `lorebooks_table`, add:

```python
groups_table = Table(
    "story_rp_groups",
    _metadata,
    Column("group_id", String(MAX_DB_KEY_LENGTH), primary_key=True),
    Column("group_json", Text, nullable=False),
    Column("updated_at", DateTime),
)
```

Change the `_SqlTables` docstring to `"""Creates the character, lorebook and group tables on first use."""`.

In the `EngineStore` docstring, change `character cards and lorebooks` to `character cards, lorebooks and groups`, and `characters and lorebooks are JSON files` to `characters, lorebooks and groups are JSON files`.

In `EngineStore.__init__`, after `self.lorebooks_dir = ...`, add:

```python
        self.groups_dir = os.path.join(storage_dir, "groups")
```

In the `if engine is None:` branch, add:

```python
            self._groups = _FileCollection(self.groups_dir, GroupCard)
```

In the `else:` branch, add:

```python
            self._groups = _SqlCollection(tables, groups_table, "group_id", "group_json", GroupCard)
```

Append these methods to `EngineStore`:

```python
    async def save_group(self, group_id: str, group: GroupCard) -> None:
        await self._groups.put(_sanitize_key(group_id), group)

    async def get_group(self, group_id: str) -> Optional[GroupCard]:
        return await self._groups.get(_sanitize_key(group_id))

    async def list_groups(self) -> Dict[str, GroupCard]:
        return await self._groups.list()

    async def delete_group(self, group_id: str) -> bool:
        """Deletes a group; returns False if it did not exist."""
        return await self._groups.delete(_sanitize_key(group_id))
```

- [ ] **Step 5: Run the tests to verify they pass.**

Run: `uv run pytest tests/story_rp_engine/test_storage.py -v`
Expected: all pass (`test_group_crud[files]` and `[sqlite]` pass, `[postgres]` skipped).

- [ ] **Step 6: Commit.**

```bash
git add src/story_rp_engine/core/types.py src/story_rp_engine/storage/store.py tests/story_rp_engine/test_storage.py
git commit -m "add group card type and storage"
```

---

### Task 2: Group prompts and agent names

**Files:**
- Create: `src/story_rp_engine/group/__init__.py` (empty)
- Create: `src/story_rp_engine/group/prompt_builder.py`
- Test: `tests/story_rp_engine/test_group_prompt_builder.py`

**Interfaces:**
- Consumes: `CharacterCard`, `GroupCard` (Task 1); `story_rp_engine.rp.prompt_builder.replace_macros(text, char_name, user_name) -> str`.
- Produces:
  - `group_agent_name(char_id: str) -> str`
  - `build_group_system_instruction(card: CharacterCard, group: GroupCard, cards: List[CharacterCard], user_name: str = "User", greeting: Optional[str] = None) -> str`. Its first sentence is `You are {card.name} in a group roleplay with ...`, and the Task 3 test fixture relies on that.
  - `build_selector_instruction(group: GroupCard, cards: List[CharacterCard], user_name: str = "User") -> str`
  - `group_summary_prompt(cards: List[CharacterCard]) -> str`: a `str.format` template with a `{conversation_history}` placeholder.

- [ ] **Step 1: Write the failing tests.** Create `tests/story_rp_engine/test_group_prompt_builder.py`:

```python
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.prompt_builder import (
    build_group_system_instruction,
    build_selector_instruction,
    group_agent_name,
    group_summary_prompt,
)

ALICE = CharacterCard(char_id="alice", name="Alice", description="Alice owes {{user}} a favor.", scenario="Alice's own scenario.")
BOB = CharacterCard(char_id="bob-2", name="Bob", description="A blacksmith.")
QI = CharacterCard(char_id="思琪", name="思琪", description="A poet.")
GROUP = GroupCard(group_id="tavern", name="Tavern", char_ids=["alice", "bob-2", "思琪"], scenario="{{user}} meets everyone at the tavern.")


def test_group_agent_name_keeps_ids_distinct():
    assert group_agent_name("alice") == "char_alice"
    assert group_agent_name("bob-2") == "char_bob_2"
    # Non-ASCII letters are valid in identifiers, so CJK IDs don't collapse to underscores.
    assert group_agent_name("思琪") == "char_思琪"
    assert group_agent_name("思琪") != group_agent_name("王五")
    assert group_agent_name("思琪").isidentifier()


def test_group_instruction_uses_group_scenario_and_introduces_the_others():
    text = build_group_system_instruction(ALICE, GROUP, [ALICE, BOB, QI], user_name="Sam", greeting="The fire crackles.")

    assert text.startswith("You are Alice in a group roleplay with Sam, Bob, 思琪.")
    assert "Alice owes Sam a favor." in text
    assert "Sam meets everyone at the tavern." in text
    assert "Alice's own scenario." not in text
    assert "- Bob (shown as [char_bob_2]): A blacksmith." in text
    assert "[char_alice]" not in text
    assert "The fire crackles." in text


def test_selector_instruction_lists_member_ids():
    text = build_selector_instruction(GROUP, [ALICE, BOB], user_name="Sam")

    assert "- alice: Alice (shown as [char_alice]). Alice owes Sam a favor." in text
    assert "- bob-2: Bob (shown as [char_bob_2]). A blacksmith." in text
    assert "Sam meets everyone at the tavern." in text
    assert '{"speakers"' in text


def test_group_summary_prompt_formats_with_braces_in_names():
    odd = CharacterCard(char_id="odd", name="{Odd}")
    filled = group_summary_prompt([ALICE, odd]).format(conversation_history="HISTORY")

    assert "- char_alice: Alice" in filled
    assert "- char_odd: {Odd}" in filled
    assert filled.endswith("HISTORY")
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `uv run pytest tests/story_rp_engine/test_group_prompt_builder.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'story_rp_engine.group'`.

- [ ] **Step 3: Implement.** Create an empty `src/story_rp_engine/group/__init__.py`. Then create `src/story_rp_engine/group/prompt_builder.py`:

```python
import re
from typing import List, Optional
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.rp.prompt_builder import replace_macros


def group_agent_name(char_id: str) -> str:
    """ADK agent name for a group member; its events are authored under this name.

    Agent names must be identifiers. str.isidentifier accepts non-ASCII letters, so only non-word characters
    are replaced and CJK IDs stay distinct.
    """
    return "char_" + re.sub(r"\W", "_", char_id)


def _brief(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_group_system_instruction(
    card: CharacterCard,
    group: GroupCard,
    cards: List[CharacterCard],
    user_name: str = "User",
    greeting: Optional[str] = None,
) -> str:
    """Builds a group member's system prompt: its own card, with the group's scenario and the other members."""
    char_name = card.name

    def fill(text: Optional[str]) -> str:
        return replace_macros(text, char_name, user_name).strip()

    others = [c for c in cards if c.char_id != card.char_id]
    company = ", ".join([user_name] + [c.name for c in others])
    sections = [
        f"You are {char_name} in a group roleplay with {company}. Write only {char_name}'s dialogue, actions and "
        f"thoughts. Never write the words, actions or decisions of {user_name} or the other characters; they speak "
        "for themselves."
    ]
    if card.system_prompt:
        sections.append(fill(card.system_prompt))

    sections.append(f"<character>\n{fill(card.description)}\n</character>")
    if card.personality:
        sections.append(f"<personality>\n{fill(card.personality)}\n</personality>")
    if group.scenario:
        sections.append(f"<scenario>\n{fill(group.scenario)}\n</scenario>")
    if others:
        cast = "\n".join(
            f"- {c.name} (shown as [{group_agent_name(c.char_id)}]): "
            f"{_brief(replace_macros(c.description, c.name, user_name))}".rstrip()
            for c in others
        )
        sections.append(
            "<other_characters>\nThe conversation shows the other characters' lines as \"[name] said:\".\n"
            f"{cast}\n</other_characters>"
        )
    if card.mes_example:
        sections.append(
            f"<example_dialogue>\n{fill(card.mes_example)}\n</example_dialogue>\n"
            "The examples show voice and style only; they are not part of the story."
        )

    sections.append(
        "Guidelines:\n"
        "- Stay in character and consistent with established facts.\n"
        "- Reply in the language the conversation uses; usually 1-2 paragraphs, since the others reply too.\n"
        f"- React to what {user_name} and the others just said; don't repeat their lines or narrate their reactions.\n"
        "- Vary your wording; don't reuse phrases or openings from earlier replies.\n"
        f"- If {user_name} writes (OOC: ...), answer briefly out of character."
    )
    if greeting:
        sections.append(
            f"<opening_message>\n{fill(greeting)}\n</opening_message>\n"
            f"This message opened the scene before {user_name}'s first message."
        )
    if card.post_history_instructions:
        sections.append(fill(card.post_history_instructions))

    return "\n\n".join(sections)


def build_selector_instruction(group: GroupCard, cards: List[CharacterCard], user_name: str = "User") -> str:
    """Builds the speaker selector's system prompt: the members, the scene, and the JSON it must return."""
    members = "\n".join(
        f"- {c.char_id}: {c.name} (shown as [{group_agent_name(c.char_id)}]). "
        f"{_brief(replace_macros(c.description, c.name, user_name))}".rstrip()
        for c in cards
    )
    sections = [f"You decide who speaks next in a group roleplay between {user_name} and these characters:\n{members}"]
    if group.scenario:
        sections.append(f"Scenario:\n{replace_macros(group.scenario, group.name, user_name).strip()}")
    sections.append(
        f"Read the conversation and choose which characters reply to {user_name}'s latest message, in speaking "
        f"order. Put first any character {user_name} addresses. Include the characters who would naturally react "
        "and leave out those with nothing to add, but choose at least one.\n"
        'Reply with JSON only, using the ids above: {"speakers": ["<id>", ...]}'
    )
    return "\n\n".join(sections)


def group_summary_prompt(cards: List[CharacterCard]) -> str:
    """Compaction prompt for a group chat. ADK fills it with str.format, so braces in names are doubled."""
    cast = "\n".join(f"- {group_agent_name(c.char_id)}: {c.name}" for c in cards)
    cast = cast.replace("{", "{{").replace("}", "}}")
    return (
        "Below is part of a group roleplay: the user's messages (user) and the characters' replies, labelled by "
        f"speaker:\n{cast}\n"
        "Ignore the speaker_selector lines; they only pick who speaks next. Summarize the scene so far for the "
        "characters who will continue it: who is present, the events in order, relationships, promises and "
        "unresolved threads, and any lasting preferences the user stated. Refer to the characters by name, keep "
        "names and concrete details exact, and write the summary in the conversation's language.\n\n"
        "{conversation_history}"
    )
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `uv run pytest tests/story_rp_engine/test_group_prompt_builder.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit.**

```bash
git add src/story_rp_engine/group/__init__.py src/story_rp_engine/group/prompt_builder.py tests/story_rp_engine/test_group_prompt_builder.py
git commit -m "add group chat prompts and agent names"
```

---

### Task 3: Group agents and workflow

**Files:**
- Create: `src/story_rp_engine/group/selector_agent.py`
- Create: `src/story_rp_engine/group/char_agent.py`
- Create: `src/story_rp_engine/group/workflow.py`
- Create: `tests/story_rp_engine/conftest.py`
- Test: `tests/story_rp_engine/test_group_workflow.py`

**Interfaces:**
- Consumes: the Task 2 functions; `rp.callbacks.rp_before_model_callback`; `core.model_provider.get_adk_model`, `get_generate_config`.
- Produces:
  - `SPEAKER_SELECTOR = "speaker_selector"` and `create_speaker_selector(group, cards, config) -> LlmAgent`, in `selector_agent.py`.
  - `create_group_char_agent(card, group, cards, config) -> LlmAgent`, in `char_agent.py`.
  - In `workflow.py`: `SCENE_BRANCH = "group"`, `pick_speakers(plan: Optional[dict], member_ids: List[str]) -> List[str]`, `group_speakers(group: GroupCard) -> Dict[str, str]` (agent name → `char_id`), and `create_group_workflow(group, cards, config) -> Workflow`.
  - The `group_models` pytest fixture, used by Tasks 4, 5 and 7. It has attributes `.plan` (a dict, or a raw string reply) and `.requests` (dict of name → list of `LlmRequest`, keyed by `"speaker_selector"` or the character's display name).

- [ ] **Step 1: Write the fixture.** Create `tests/story_rp_engine/conftest.py`:

```python
import json
import re
from types import SimpleNamespace
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from story_rp_engine.group import char_agent, selector_agent


def _reply(text, partial=False):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]), partial=partial)


@pytest.fixture
def group_models(monkeypatch):
    """Fake models for group chat agents.

    The selector replies with `group_models.plan` (a dict, or a raw string). Each character replies
    "<Name> line", streamed as "<Name> " + "line", and reads its name from its system prompt. Requests are
    recorded in `group_models.requests`, keyed by "speaker_selector" or the character's name.
    """
    fake = SimpleNamespace(plan={"speakers": []}, requests={})

    class SelectorLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            fake.requests.setdefault("speaker_selector", []).append(llm_request)
            yield _reply(fake.plan if isinstance(fake.plan, str) else json.dumps(fake.plan))

    class CharacterLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            name = re.match(r"You are (\S+) in a group roleplay", llm_request.config.system_instruction).group(1)
            fake.requests.setdefault(name, []).append(llm_request)
            if stream:
                yield _reply(f"{name} ", partial=True)
                yield _reply("line", partial=True)
            yield _reply(f"{name} line")

    monkeypatch.setattr(selector_agent, "get_adk_model", lambda config: SelectorLlm())
    monkeypatch.setattr(char_agent, "get_adk_model", lambda config: CharacterLlm())
    return fake
```

- [ ] **Step 2: Write the failing tests.** Create `tests/story_rp_engine/test_group_workflow.py`:

```python
import pytest
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, GroupCard, Lorebook, LorebookEntry
from story_rp_engine.group.workflow import create_group_workflow, group_speakers, pick_speakers

ALICE = CharacterCard(char_id="alice", name="Alice", description="A bard.")
BOB = CharacterCard(char_id="bob", name="Bob", description="A blacksmith.")
GROUP = GroupCard(group_id="tavern", name="Tavern", char_ids=["alice", "bob"])


def _texts(llm_request):
    return ["".join(p.text or "" for p in c.parts) for c in llm_request.contents]


async def _run_turns(messages, state_delta=None):
    """Runs group turns in one session and returns the authors of its events."""
    service = InMemorySessionService()
    runner = Runner(
        app=App(name="group_app", root_agent=create_group_workflow(GROUP, [ALICE, BOB], EngineConfig())),
        session_service=service,
        auto_create_session=True,
    )
    for message in messages:
        async for _ in runner.run_async(
            user_id="User",
            session_id="s1",
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=message)]),
            state_delta=state_delta,
        ):
            pass
    session = await service.get_session(app_name="group_app", user_id="User", session_id="s1")
    return [ev.author for ev in session.events]


def test_pick_speakers_drops_repeats_and_falls_back_to_group_order():
    assert pick_speakers({"speakers": ["bob", "alice", "bob"]}, ["alice", "bob"]) == ["bob", "alice"]
    assert pick_speakers({"speakers": []}, ["alice", "bob"]) == ["alice", "bob"]
    assert pick_speakers(None, ["alice", "bob"]) == ["alice", "bob"]


def test_group_speakers_maps_agent_names_to_char_ids():
    assert group_speakers(GROUP) == {"char_alice": "alice", "char_bob": "bob"}


@pytest.mark.anyio
async def test_characters_reply_in_selected_order(group_models):
    group_models.plan = {"speakers": ["bob", "alice"]}
    assert await _run_turns(["Hello, everyone."]) == ["user", "speaker_selector", "char_bob", "char_alice"]


@pytest.mark.anyio
async def test_characters_see_each_other_but_not_the_selector(group_models):
    group_models.plan = {"speakers": ["bob", "alice"]}
    await _run_turns(["Hello, everyone.", "How was the forge?"])

    assert _texts(group_models.requests["Alice"][0]) == ["Hello, everyone.", "For context:[char_bob] said: Bob line"]
    assert _texts(group_models.requests["Bob"][1]) == [
        "Hello, everyone.",
        "Bob line",
        "For context:[char_alice] said: Alice line",
        "How was the forge?",
    ]
    # The selector reads the characters' lines.
    assert "For context:[char_alice] said: Alice line" in _texts(group_models.requests["speaker_selector"][1])


@pytest.mark.anyio
async def test_selector_schema_allows_only_members(group_models):
    group_models.plan = {"speakers": ["alice"]}
    await _run_turns(["Hi."])

    schema = group_models.requests["speaker_selector"][0].config.response_schema
    assert schema.model_validate({"speakers": ["bob"]}).speakers == ["bob"]
    with pytest.raises(ValidationError):
        schema.model_validate({"speakers": ["zed"]})


@pytest.mark.anyio
async def test_characters_get_lore_and_authors_note(group_models):
    group_models.plan = {"speakers": ["alice"]}
    lorebook = Lorebook(name="World", entries=[LorebookEntry(keys=["forge"], content="The forge never cools.")])
    await _run_turns(["Is the forge lit?"], state_delta={"lorebook": lorebook.model_dump(), "authors_note": "Keep it short."})

    latest = _texts(group_models.requests["Alice"][0])[-1]
    assert latest.startswith("Is the forge lit?")
    assert "The forge never cools." in latest
    assert "Author's note: Keep it short." in latest
```

- [ ] **Step 3: Run the tests to verify they fail.**

Run: `uv run pytest tests/story_rp_engine/test_group_workflow.py -v`
Expected: errors such as `ImportError: cannot import name 'char_agent' from 'story_rp_engine.group'` (from conftest) and `ModuleNotFoundError: ... group.workflow`.

- [ ] **Step 4: Implement the selector.** Create `src/story_rp_engine/group/selector_agent.py`:

```python
from typing import List, Literal
from google.adk.agents import LlmAgent
from pydantic import create_model
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.prompt_builder import build_selector_instruction

SPEAKER_SELECTOR = "speaker_selector"


def create_speaker_selector(group: GroupCard, cards: List[CharacterCard], config: EngineConfig) -> LlmAgent:
    """An agent that reads the conversation and returns {"speakers": [char_id, ...]}.

    The schema lists the member IDs, so backends with constrained decoding (Gemini, llama.cpp) can only
    produce members, and any other ID fails validation.
    """
    member_ids = tuple(c.char_id for c in cards)
    speaker_plan = create_model("SpeakerPlan", speakers=(List[Literal[member_ids]], ...))
    return LlmAgent(
        name=SPEAKER_SELECTOR,
        model=get_adk_model(config),
        # Callable, like the RP agent's: card text may contain braces such as {{user}}.
        instruction=lambda ctx: build_selector_instruction(group, cards, user_name=ctx.state.get("user_name") or "User"),
        generate_content_config=get_generate_config(config),
        # Workflow agents default to no history; the selector reads the whole (compacted) conversation.
        include_contents="default",
        output_schema=speaker_plan,
    )
```

- [ ] **Step 5: Implement the character agent.** Create `src/story_rp_engine/group/char_agent.py`:

```python
from typing import List
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.prompt_builder import build_group_system_instruction, group_agent_name
from story_rp_engine.rp.callbacks import rp_before_model_callback


def create_group_char_agent(
    card: CharacterCard,
    group: GroupCard,
    cards: List[CharacterCard],
    config: EngineConfig,
) -> LlmAgent:
    """A group member's agent: it reads the whole conversation and writes only its own character's reply."""
    return LlmAgent(
        name=group_agent_name(card.char_id),
        model=get_adk_model(config),
        # Callable, like the RP agent's: card text may contain braces such as {{user}}.
        instruction=lambda ctx: build_group_system_instruction(
            card, group, cards, user_name=ctx.state.get("user_name") or "User", greeting=ctx.state.get("greeting")
        ),
        generate_content_config=get_generate_config(config),
        include_contents="default",
        before_model_callback=rp_before_model_callback,
    )
```

- [ ] **Step 6: Implement the workflow.** Create `src/story_rp_engine/group/workflow.py`:

```python
from typing import Dict, List, Optional
from google.adk import Context, Workflow
from google.adk.workflow import node
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.char_agent import create_group_char_agent
from story_rp_engine.group.prompt_builder import group_agent_name
from story_rp_engine.group.selector_agent import create_speaker_selector

# Characters run on this branch and the selector on a sub-branch of it. ADK shows an agent the events of its own
# branch and its ancestors, so the selector sees the characters' lines but the characters never see its JSON.
# A node with no branch would see everything, so the characters need the named branch too.
SCENE_BRANCH = "group"


def pick_speakers(plan: Optional[dict], member_ids: List[str]) -> List[str]:
    """The selector's speakers without repeats, or every member in group order if it picked no one."""
    speakers = list(dict.fromkeys((plan or {}).get("speakers", [])))
    return speakers or list(member_ids)


def group_speakers(group: GroupCard) -> Dict[str, str]:
    """Maps each member's agent name (the author of its events) to its char_id."""
    return {group_agent_name(char_id): char_id for char_id in group.char_ids}


def create_group_workflow(group: GroupCard, cards: List[CharacterCard], config: EngineConfig) -> Workflow:
    """One group chat turn: the speaker selector picks who replies, then each chosen character replies in order,
    seeing the user's message and the replies before it."""
    selector = create_speaker_selector(group, cards, config)
    agents = {c.char_id: create_group_char_agent(c, group, cards, config) for c in cards}

    @node(rerun_on_resume=True)
    async def group_turn(ctx: Context):
        plan = await ctx.run_node(selector, use_sub_branch=True, override_branch=SCENE_BRANCH)
        for char_id in pick_speakers(plan, list(agents)):
            await ctx.run_node(agents[char_id], override_branch=SCENE_BRANCH)

    return Workflow(name="group_workflow", edges=[("START", group_turn)])
```

- [ ] **Step 7: Run the tests to verify they pass.**

Run: `uv run pytest tests/story_rp_engine/test_group_workflow.py tests/story_rp_engine/test_group_prompt_builder.py -v`
Expected: all pass.

- [ ] **Step 8: Commit.**

```bash
git add src/story_rp_engine/group tests/story_rp_engine/conftest.py tests/story_rp_engine/test_group_workflow.py
git commit -m "add group chat workflow with speaker selector"
```

---

### Task 4: Group runner in AgentRegistry

**Files:**
- Modify: `src/story_rp_engine/core/agent_registry.py`
- Test: `tests/story_rp_engine/test_agent_registry.py`

**Interfaces:**
- Consumes: `create_group_workflow`, `group_summary_prompt`; `EngineStore.get_group`, `get_character`.
- Produces: `AgentRegistry.get_or_create_group_runner(group_id: str) -> Runner`, which raises `ValueError("Group <id> not found")` or `ValueError("Group <id> has no characters")`. Also `AgentRegistry.forget_group(group_id: str) -> None`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/story_rp_engine/test_agent_registry.py`, and add `GroupCard` to its types import:

```python
async def _group_store(tmp_path, char_ids):
    store = EngineStore(storage_dir=str(tmp_path))
    for char_id in ("alice", "bob"):
        await store.save_character(char_id, CharacterCard(char_id=char_id, name=char_id.title()))
    await store.save_group("tavern", GroupCard(group_id="tavern", name="Tavern", char_ids=char_ids))
    return store


@pytest.mark.anyio
async def test_group_runner_cached_and_rebuilt_when_a_member_changes(tmp_path):
    store = await _group_store(tmp_path, ["alice", "bob"])
    registry = AgentRegistry(config=EngineConfig(), store=store)

    runner = await registry.get_or_create_group_runner("tavern")
    assert runner.app_name == "group_app"
    assert await registry.get_or_create_group_runner("tavern") is runner

    await store.save_character("bob", CharacterCard(char_id="bob", name="Bob", description="Now a smith."))
    rebuilt = await registry.get_or_create_group_runner("tavern")
    assert rebuilt is not runner

    registry.forget_group("tavern")
    assert await registry.get_or_create_group_runner("tavern") is not rebuilt


@pytest.mark.anyio
async def test_group_runner_skips_deleted_members(tmp_path, group_models):
    from google.genai import types

    store = await _group_store(tmp_path, ["alice", "ghost"])
    runner = await AgentRegistry(config=EngineConfig(), store=store).get_or_create_group_runner("tavern")

    # The selector picks no one, so every remaining member speaks.
    authors = [
        ev.author
        async for ev in runner.run_async(
            user_id="User", session_id="s1", new_message=types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
        )
    ]
    assert authors == ["speaker_selector", "char_alice"]


@pytest.mark.anyio
async def test_group_runner_errors(tmp_path):
    store = await _group_store(tmp_path, ["ghost"])
    registry = AgentRegistry(config=EngineConfig(), store=store)

    with pytest.raises(ValueError, match="Group missing not found"):
        await registry.get_or_create_group_runner("missing")
    with pytest.raises(ValueError, match="Group tavern has no characters"):
        await registry.get_or_create_group_runner("tavern")
```

Note: `runner.run_async` doesn't yield the user's own message event, so the authors list starts at the selector.

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `uv run pytest tests/story_rp_engine/test_agent_registry.py -k group -v`
Expected: FAIL with `AttributeError: 'AgentRegistry' object has no attribute 'get_or_create_group_runner'`.

- [ ] **Step 3: Implement.** In `src/story_rp_engine/core/agent_registry.py`:

Change the typing import to `from typing import Dict, List, Optional, Tuple`, and the types import to `from story_rp_engine.core.types import CharacterCard, GroupCard, Lorebook`.

In `__init__`, after `self._story_runner = None`, add:

```python
        # Group runners with the (group, member cards) they were built from.
        self._group_runners: Dict[str, Tuple[Tuple[GroupCard, List[CharacterCard]], Runner]] = {}
```

After `_build_compaction_config`, add:

```python
    def _turn_count_compaction_config(self, prompt_template: str) -> Optional[EventsCompactionConfig]:
        """Compaction for workflow apps: turn-count only. Token-threshold compaction also runs before each model
        call, so it could fire between two agents of one turn and summarize away what the first one wrote."""
        if not self.config.compaction_enabled:
            return None
        return EventsCompactionConfig(
            compaction_interval=self.config.compaction_interval,
            overlap_size=self.config.compaction_overlap_size,
            summarizer=LlmEventSummarizer(
                llm=get_adk_model(self.config),
                prompt_template=self.config.compaction_prompt_template or prompt_template,
            ),
        )
```

In `get_story_runner`, replace the whole block from `compaction_config = None` through the closing `)` of the `if self.config.compaction_enabled:` block with:

```python
            compaction_config = self._turn_count_compaction_config(STORY_SUMMARY_PROMPT)
```

Append the group methods at the end of the class:

```python
    async def get_or_create_group_runner(self, group_id: str) -> Runner:
        """Retrieves or creates the ADK Runner for a group chat.

        The store may be shared with other instances, so the group and its member cards are re-read each time and
        the runner is rebuilt when they changed. Members whose cards were deleted are left out.
        """
        group = await self.store.get_group(group_id)
        if group is None:
            self.forget_group(group_id)
            raise ValueError(f"Group {group_id} not found")
        cards = []
        for char_id in group.char_ids:
            card = await self.store.get_character(char_id)
            if card is not None:
                cards.append(card)
        if not cards:
            raise ValueError(f"Group {group_id} has no characters")

        cached = self._group_runners.get(group_id)
        if cached is not None and cached[0] == (group, cards):
            return cached[1]

        from story_rp_engine.group.prompt_builder import group_summary_prompt
        from story_rp_engine.group.workflow import create_group_workflow
        app = App(
            name="group_app",
            root_agent=create_group_workflow(group, cards, self.config),
            events_compaction_config=self._turn_count_compaction_config(group_summary_prompt(cards)),
        )
        runner = Runner(app=app, session_service=self.store.session_service, auto_create_session=True)
        self._group_runners[group_id] = ((group, cards), runner)
        return runner

    def forget_group(self, group_id: str) -> None:
        """Drops the cached runner for a group (e.g. after deletion)."""
        self._group_runners.pop(group_id, None)
```

- [ ] **Step 4: Run the tests to verify they pass, including the story compaction tests touched by the helper.**

Run: `uv run pytest tests/story_rp_engine/test_agent_registry.py tests/story_rp_engine/test_story_workflow.py tests/story_rp_engine/test_compaction.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/story_rp_engine/core/agent_registry.py tests/story_rp_engine/test_agent_registry.py
git commit -m "cache group chat runners in the agent registry"
```

---

### Task 5: Streaming group turns

**Files:**
- Modify: `src/story_rp_engine/core/agent_utils.py`
- Test: `tests/story_rp_engine/test_agent_utils.py`

**Interfaces:**
- Consumes: `group_speakers`, `create_group_workflow`, the `group_models` fixture.
- Produces:
  - `stream_group_turn(runner, user_id, session_id, message, speakers: Dict[str, str], state_delta=None) -> AsyncIterator[Tuple[str, str]]`
  - `format_sse_stream(generator, chunk_size)` also accepts `(speaker, text)` tuples, emitting the group SSE format from Global Constraints.

- [ ] **Step 1: Write the failing tests.** Append to `tests/story_rp_engine/test_agent_utils.py`:

```python
@pytest.mark.anyio
async def test_format_sse_stream_labels_group_replies_by_speaker():
    async def mock_gen():
        yield ("bob", "Bob ")
        yield ("bob", "line")
        yield ("alice", "Alice line")

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=16)]
    assert events == [
        'data: {"speaker": "bob", "delta": "Bob line"}\n\n',
        'data: {"speaker": "alice", "delta": "Alice line"}\n\n',
        'data: {"replies": [{"speaker": "bob", "text": "Bob line"}, {"speaker": "alice", "text": "Alice line"}], "done": true}\n\n',
        "data: [DONE]\n\n",
    ]


@pytest.mark.anyio
async def test_stream_group_turn_yields_speaker_chunks_without_the_selector(group_models):
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from story_rp_engine.core.agent_utils import stream_group_turn
    from story_rp_engine.core.config import EngineConfig
    from story_rp_engine.core.types import CharacterCard, GroupCard
    from story_rp_engine.group.workflow import create_group_workflow, group_speakers

    group_models.plan = {"speakers": ["bob", "alice"]}
    group = GroupCard(group_id="g", name="G", char_ids=["alice", "bob"])
    cards = [CharacterCard(char_id="alice", name="Alice"), CharacterCard(char_id="bob", name="Bob")]
    runner = Runner(
        app=App(name="group_app", root_agent=create_group_workflow(group, cards, EngineConfig())),
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    chunks = [c async for c in stream_group_turn(runner, "User", "s1", "Hi.", speakers=group_speakers(group))]
    assert chunks == [("bob", "Bob "), ("bob", "line"), ("alice", "Alice "), ("alice", "line")]
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `uv run pytest tests/story_rp_engine/test_agent_utils.py -v`
Expected: the first new test fails (tuples are treated as strings: `AttributeError: 'tuple' object has no attribute 'strip'`). The second fails with `ImportError: cannot import name 'stream_group_turn'`.

- [ ] **Step 3: Implement.** In `src/story_rp_engine/core/agent_utils.py`:

Change the typing import to `from typing import AsyncIterator, Dict, Optional, Tuple, Union`.

After `stream_runner_turn`, add:

```python
async def stream_group_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
    speakers: Dict[str, str],
    state_delta: Optional[dict] = None,
) -> AsyncIterator[Tuple[str, str]]:
    """Streams a group chat turn as (char_id, text) chunks.

    `speakers` maps agent names to char_ids; events from other authors (the speaker selector) are skipped.
    """
    content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
    run_cfg = RunConfig(streaming_mode=StreamingMode.SSE)
    streamed = set()  # speakers whose reply already arrived in partial chunks

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        state_delta=state_delta,
        run_config=run_cfg,
    ):
        char_id = speakers.get(event.author)
        if char_id is None or not event.content or not event.content.parts:
            continue
        text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
        if not text:
            continue
        if event.partial:
            streamed.add(char_id)
            yield char_id, text
        elif char_id not in streamed:
            yield char_id, text
```

Replace the whole `format_sse_stream` function with:

```python
async def format_sse_stream(
    generator: AsyncIterator[Union[str, Tuple[str, str]]],
    chunk_size: Optional[int] = 4,
) -> AsyncIterator[str]:
    """Buffers string chunks, yielding structured SSE JSON deltas and a final complete text.

    Group chat streams (speaker, text) pairs instead: each delta then carries its speaker, a new speaker flushes
    the buffer, and the final event lists each speaker's reply in place of full_text.
    """
    buffer = []
    full_text_chunks = []
    replies = []  # [{"speaker", "text"}], filled from (speaker, text) chunks
    chunk_threshold = max(1, chunk_size or 4)
    buffered_tokens = 0

    def delta_event() -> str:
        data = {"delta": "".join(buffer)}
        if replies:
            data = {"speaker": replies[-1]["speaker"], **data}
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        async for chunk in generator:
            if isinstance(chunk, tuple):
                speaker, chunk = chunk
                if chunk and (not replies or replies[-1]["speaker"] != speaker):
                    if buffer:
                        yield delta_event()
                        buffer.clear()
                        buffered_tokens = 0
                    replies.append({"speaker": speaker, "text": ""})
                if chunk:
                    replies[-1]["text"] += chunk
            if not chunk:
                continue
            buffer.append(chunk)
            full_text_chunks.append(chunk)

            words = chunk.strip().split()
            chunk_tokens = len(words) if words else 1
            buffered_tokens += chunk_tokens

            if chunk_threshold <= 1 or buffered_tokens >= chunk_threshold or len(buffer) >= chunk_threshold or "\n" in chunk:
                yield delta_event()
                buffer.clear()
                buffered_tokens = 0
    except Exception as e:
        # The 200 response has already started, so report model errors (e.g. Gemini API errors) in-stream.
        logger.exception("Model turn failed")
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if buffer:
        yield delta_event()
        buffer.clear()

    if replies:
        done = {"replies": replies, "done": True}
    else:
        done = {"full_text": "".join(full_text_chunks), "done": True}
    yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
```

- [ ] **Step 4: Run the tests to verify they pass, including the RP/story SSE tests that pin the old format.**

Run: `uv run pytest tests/story_rp_engine/test_agent_utils.py tests/story_rp_engine/test_api.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/story_rp_engine/core/agent_utils.py tests/story_rp_engine/test_agent_utils.py
git commit -m "stream group chat turns with speaker-labelled SSE deltas"
```

---

### Task 6: Shared chat session helpers

This is a pure refactor: the RP routes' behavior and responses must not change. The existing tests are the safety net.

**Files:**
- Create: `src/story_rp_engine/api/chat_sessions.py`
- Modify: `src/story_rp_engine/api/routes_rp.py`
- Test: existing `tests/story_rp_engine/test_api.py`, `tests/test_frontend_integration.py`, `tests/test_web_mount.py`

**Interfaces:**
- Produces, in `story_rp_engine.api.chat_sessions`:
  - `USER_ID = "User"`
  - `RenameSessionRequest(title: str)` and `DeleteTurnRequest(turn_index: int, truncate_subsequent: bool = False)`
  - `checked_session_id(session_id) -> str` (raises `HTTPException(400)`)
  - `chat_state_delta(req, request, **owner) -> dict`
  - `message_event_indexes(events, skip_authors=()) -> List[int]`
  - `list_chat_sessions(session_service, app_name, owner_key, owner_id) -> list[dict]`
  - `rename_chat(session_service, app_name, session_id, title) -> None` (raises `HTTPException(404)`)
  - `delete_chat_turn(session_service, app_name, session_id, turn_index, truncate_subsequent, skip_authors=()) -> int`

- [ ] **Step 1: Record the baseline.**

Run: `uv run pytest tests/story_rp_engine/test_api.py tests/test_frontend_integration.py tests/test_web_mount.py -q`
Expected: all pass. Note the count.

- [ ] **Step 2: Create the helpers.** Create `src/story_rp_engine/api/chat_sessions.py`:

```python
"""Chat session helpers shared by the roleplay and group chat routes; each router passes its own ADK app name."""

from typing import Collection, List
from fastapi import HTTPException, Request
from google.adk.events import Event, EventActions
from pydantic import BaseModel
from story_rp_engine.storage.store import _sanitize_key

# Every chat session uses this user ID; the display name lives in session state.
USER_ID = "User"


class RenameSessionRequest(BaseModel):
    title: str


class DeleteTurnRequest(BaseModel):
    turn_index: int
    truncate_subsequent: bool = False


def checked_session_id(session_id: str) -> str:
    try:
        return _sanitize_key(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def chat_state_delta(req, request: Request, **owner) -> dict:
    """Session state changes for a chat turn; a lorebook is copied in only when lorebook_id is sent.

    `owner` (char_id=... or group_id=...) and last_message are read by the history list, which gets session
    state but no events.
    """
    state_delta = {
        "authors_note": req.authors_note,
        "user_name": req.user_name or "User",
        "greeting": req.greeting,
        **owner,
        "last_message": req.message[:80],
    }
    if req.lorebook_id is not None:
        lorebook = None
        if req.lorebook_id:
            lorebook = await request.app.state.store.get_lorebook(req.lorebook_id)
            if lorebook is None:
                raise HTTPException(status_code=404, detail="Lorebook not found")
        state_delta["lorebook"] = lorebook.model_dump() if lorebook else None
        # Read by the history list, so reopening a chat reselects its lorebook.
        state_delta["lorebook_id"] = req.lorebook_id or None
    return state_delta


def message_event_indexes(events, skip_authors: Collection[str] = ()) -> List[int]:
    """Indexes of the events the chat shows: user and model text, without ADK compaction summaries or the events
    of skip_authors.

    Compaction only adds a summary event and keeps the raw events, so the full history is still there.
    """
    return [
        i
        for i, ev in enumerate(events)
        if not (ev.actions and ev.actions.compaction)
        and ev.author not in skip_authors
        and ev.content
        and any(p.text for p in ev.content.parts or [])
    ]


async def list_chat_sessions(session_service, app_name: str, owner_key: str, owner_id: str) -> list:
    """Chats whose state[owner_key] is owner_id (e.g. a character's chats), newest first."""
    response = await session_service.list_sessions(app_name=app_name, user_id=USER_ID)
    return [
        {
            "session_id": s.id,
            "updated_at": s.last_update_time,
            "title": s.state.get("title"),
            "last_message": s.state.get("last_message", ""),
            "greeting": s.state.get("greeting"),
            "user_name": s.state.get("user_name"),
            "authors_note": s.state.get("authors_note"),
            "lorebook_id": s.state.get("lorebook_id"),
        }
        for s in sorted(response.sessions, key=lambda s: s.last_update_time, reverse=True)
        if s.state.get(owner_key) == owner_id
    ]


async def rename_chat(session_service, app_name: str, session_id: str, title: str) -> None:
    session = await session_service.get_session(app_name=app_name, user_id=USER_ID, session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # A state-only event: ADK has no other way to update session state, and having no content
    # keeps it out of the chat and the prompt.
    await session_service.append_event(
        session, Event(author="user", actions=EventActions(state_delta={"title": title.strip()}))
    )


async def delete_chat_turn(
    session_service,
    app_name: str,
    session_id: str,
    turn_index: int,
    truncate_subsequent: bool,
    skip_authors: Collection[str] = (),
) -> int:
    """Deletes one shown message, or it and everything after it, by rebuilding the session; returns the number
    of events left.

    turn_index counts the messages the chat shows (see message_event_indexes), not raw events.
    """
    session = await session_service.get_session(app_name=app_name, user_id=USER_ID, session_id=session_id)
    if not session:
        return 0

    events = list(session.events)
    message_indexes = message_event_indexes(events, skip_authors)
    if truncate_subsequent:
        if 0 <= turn_index < len(message_indexes):
            events = events[: message_indexes[turn_index]]
        elif turn_index < 0:
            events = []
    elif 0 <= turn_index < len(message_indexes):
        events.pop(message_indexes[turn_index])

    await session_service.delete_session(app_name=app_name, user_id=USER_ID, session_id=session_id)
    new_session = await session_service.create_session(
        app_name=app_name, user_id=USER_ID, session_id=session_id, state=session.state
    )
    for ev in events:
        await session_service.append_event(new_session, ev)
    return len(events)
```

- [ ] **Step 3: Switch the RP routes to the helpers.** In `src/story_rp_engine/api/routes_rp.py`:

Replace the imports at the top of the file (everything up to and including `router = APIRouter(...)`) with:

```python
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.api.chat_sessions import (
    USER_ID,
    DeleteTurnRequest,
    RenameSessionRequest,
    chat_state_delta,
    checked_session_id,
    delete_chat_turn,
    list_chat_sessions,
    message_event_indexes,
    rename_chat,
)
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    format_sse_stream,
    stream_runner_turn,
)
from story_rp_engine.core.types import CharacterCard, RPChatRequest
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Roleplay"])

RP_APP = "rp_app"
```

Delete the whole `_rp_state_delta` function.

In both `chat_rp` and `chat_rp_stream`, replace `state_delta = await _rp_state_delta(req, request)` with:

```python
        state_delta = await chat_state_delta(req, request, char_id=req.char_id)
```

Replace everything from `class RenameSessionRequest(BaseModel):` to the end of the file with:

```python
@router.get("/rp/sessions")
async def list_rp_sessions(char_id: str, request: Request):
    """Chats with one character, newest first."""
    return await list_chat_sessions(request.app.state.session_service, RP_APP, "char_id", char_id)


@router.get("/rp/sessions/{session_id}/turns")
async def get_session_turns(session_id: str, request: Request):
    session = await request.app.state.session_service.get_session(
        app_name=RP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    if not session:
        return {"turns": []}
    turns = []
    for pos, idx in enumerate(message_event_indexes(session.events)):
        content = session.events[idx].content
        turns.append({"index": pos, "role": content.role, "text": "".join(p.text for p in content.parts if p.text)})
    return {"turns": turns}


@router.patch("/rp/sessions/{session_id}")
async def rename_rp_session(session_id: str, req: RenameSessionRequest, request: Request):
    await rename_chat(request.app.state.session_service, RP_APP, session_id, req.title)
    return {"status": "renamed", "session_id": session_id}


@router.delete("/rp/sessions/{session_id}")
async def clear_session(session_id: str, request: Request):
    await request.app.state.session_service.delete_session(
        app_name=RP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    return {"status": "deleted", "session_id": session_id}


@router.post("/rp/sessions/{session_id}/turns/delete")
async def delete_session_turn(session_id: str, req: DeleteTurnRequest, request: Request):
    remaining = await delete_chat_turn(
        request.app.state.session_service,
        RP_APP,
        checked_session_id(session_id),
        req.turn_index,
        req.truncate_subsequent,
    )
    return {"status": "ok", "remaining_turns": remaining}
```

- [ ] **Step 4: Run the tests to verify nothing changed.**

Run: `uv run pytest tests/story_rp_engine/test_api.py tests/test_frontend_integration.py tests/test_web_mount.py -q`
Expected: the same count as Step 1, all passing.

- [ ] **Step 5: Commit.**

```bash
git add src/story_rp_engine/api/chat_sessions.py src/story_rp_engine/api/routes_rp.py
git commit -m "move rp session routes into shared chat session helpers"
```

---

### Task 7: Group chat API and docs

**Files:**
- Create: `src/story_rp_engine/api/routes_group.py`
- Modify: `src/story_rp_engine/api/app.py`
- Modify: `CLAUDE.md`, `.agents/skills/character-rp/SKILL.md`
- Test: `tests/story_rp_engine/test_group_api.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces the HTTP API used by the UI:
  - `POST /api/v1/groups` → `{"status": "saved", "group_id"}`
  - `GET /api/v1/groups` → `{group_id: GroupCard}`
  - `GET /api/v1/groups/{id}` → `GroupCard`, or 404 `"Group not found"`
  - `DELETE /api/v1/groups/{id}` → `{"status": "deleted", "group_id"}`
  - `POST /api/v1/group/chat/stream` (body `GroupChatRequest`) → group SSE
  - `GET /api/v1/group/sessions?group_id=` → session list
  - `GET /api/v1/group/sessions/{id}/turns` → `{"turns": [{"index", "role", "speaker", "text"}]}`
  - `PATCH /api/v1/group/sessions/{id}` (`{"title"}`)
  - `DELETE /api/v1/group/sessions/{id}`
  - `POST /api/v1/group/sessions/{id}/turns/delete` (`{"turn_index", "truncate_subsequent"}`)

- [ ] **Step 1: Write the failing tests.** Create `tests/story_rp_engine/test_group_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, Lorebook, LorebookEntry
from story_rp_engine.storage.store import EngineStore

TAVERN = {
    "group_id": "tavern",
    "name": "Tavern",
    "char_ids": ["alice", "bob"],
    "scenario": "A rainy night.",
    "first_mes": "The door swings open.",
}


async def _group_client(tmp_path):
    """App with characters alice and bob, the group 'tavern', and a lorebook keyed on 'forge'."""
    store = EngineStore(storage_dir=str(tmp_path))
    for char_id, name in [("alice", "Alice"), ("bob", "Bob")]:
        await store.save_character(char_id, CharacterCard(char_id=char_id, name=name))
    await store.save_lorebook(
        "world", Lorebook(name="World", entries=[LorebookEntry(keys=["forge"], content="The forge never cools.")])
    )
    client = TestClient(create_app(store=store, config=EngineConfig(compaction_enabled=False)))
    assert client.post("/api/v1/groups", json=TAVERN).status_code == 200
    return client


def _chat(client, message, **extra):
    body = {"group_id": "tavern", "session_id": "s1", "message": message, "chunk_size": 1, **extra}
    return client.post("/api/v1/group/chat/stream", json=body)


def test_group_crud(tmp_path):
    client = TestClient(create_app(store=EngineStore(storage_dir=str(tmp_path)), config=EngineConfig()))

    assert client.post("/api/v1/groups", json=TAVERN).json() == {"status": "saved", "group_id": "tavern"}
    assert client.get("/api/v1/groups/tavern").json()["char_ids"] == ["alice", "bob"]
    assert list(client.get("/api/v1/groups").json()) == ["tavern"]
    assert client.post("/api/v1/groups", json={**TAVERN, "group_id": "../evil"}).status_code == 400

    assert client.delete("/api/v1/groups/tavern").json() == {"status": "deleted", "group_id": "tavern"}
    assert client.get("/api/v1/groups/tavern").status_code == 404
    assert client.delete("/api/v1/groups/tavern").status_code == 404


@pytest.mark.anyio
async def test_group_chat_streams_replies_by_speaker(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["bob", "alice"]}

    res = _chat(client, "Hello!")
    assert res.status_code == 200
    assert res.text == (
        'data: {"speaker": "bob", "delta": "Bob "}\n\n'
        'data: {"speaker": "bob", "delta": "line"}\n\n'
        'data: {"speaker": "alice", "delta": "Alice "}\n\n'
        'data: {"speaker": "alice", "delta": "line"}\n\n'
        'data: {"replies": [{"speaker": "bob", "text": "Bob line"}, {"speaker": "alice", "text": "Alice line"}], "done": true}\n\n'
        "data: [DONE]\n\n"
    )


@pytest.mark.anyio
async def test_group_chat_loads_lorebook_scenario_and_greeting(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["alice"]}

    assert _chat(client, "Is the forge lit?", lorebook_id="world", greeting="The door swings open.").status_code == 200
    request = group_models.requests["Alice"][-1]
    assert "The forge never cools." in "".join(p.text for p in request.contents[-1].parts)
    assert "The door swings open." in request.config.system_instruction
    assert "A rainy night." in request.config.system_instruction


@pytest.mark.anyio
async def test_group_turns_name_speakers_and_hide_the_selector(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["bob", "alice"]}
    _chat(client, "Hello!", greeting="The door swings open.")

    turns = client.get("/api/v1/group/sessions/s1/turns").json()["turns"]
    assert [(t["speaker"], t["text"]) for t in turns] == [(None, "Hello!"), ("bob", "Bob line"), ("alice", "Alice line")]

    sessions = client.get("/api/v1/group/sessions", params={"group_id": "tavern"}).json()
    assert [(s["session_id"], s["last_message"], s["greeting"]) for s in sessions] == [
        ("s1", "Hello!", "The door swings open.")
    ]
    assert client.get("/api/v1/group/sessions", params={"group_id": "other"}).json() == []


@pytest.mark.anyio
async def test_group_session_rename_delete_turn_and_delete(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["bob", "alice"]}
    _chat(client, "Hello!")

    assert client.patch("/api/v1/group/sessions/s1", json={"title": "Rainy night"}).status_code == 200
    # Index 1 is Bob's reply: the selector's hidden event doesn't count.
    client.post("/api/v1/group/sessions/s1/turns/delete", json={"turn_index": 1})
    assert [t["text"] for t in client.get("/api/v1/group/sessions/s1/turns").json()["turns"]] == ["Hello!", "Alice line"]
    assert client.get("/api/v1/group/sessions", params={"group_id": "tavern"}).json()[0]["title"] == "Rainy night"

    assert client.delete("/api/v1/group/sessions/s1").status_code == 200
    assert client.get("/api/v1/group/sessions", params={"group_id": "tavern"}).json() == []


@pytest.mark.anyio
async def test_group_chat_rejects_bad_requests(tmp_path, group_models):
    client = await _group_client(tmp_path)
    client.post("/api/v1/groups", json={**TAVERN, "group_id": "ghosts", "char_ids": ["ghost"]})

    assert _chat(client, "Hi", group_id="missing").status_code == 404
    res = _chat(client, "Hi", group_id="ghosts")
    assert res.status_code == 400
    assert "has no characters" in res.json()["detail"]
    assert _chat(client, "Hi", session_id="../evil").status_code == 400
    assert _chat(client, "Hi", lorebook_id="missing").status_code == 404


@pytest.mark.anyio
async def test_group_chat_reports_unusable_selector_reply(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = "not json"

    res = _chat(client, "Hello!")
    assert res.status_code == 200
    assert 'data: {"error": ' in res.text
    assert res.text.endswith("data: [DONE]\n\n")
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `uv run pytest tests/story_rp_engine/test_group_api.py -v`
Expected: FAIL. The routes don't exist yet, so `POST /api/v1/groups` falls through to the static UI mount and returns 405, not 200.

- [ ] **Step 3: Implement the router.** Create `src/story_rp_engine/api/routes_group.py`:

```python
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.api.chat_sessions import (
    USER_ID,
    DeleteTurnRequest,
    RenameSessionRequest,
    chat_state_delta,
    checked_session_id,
    delete_chat_turn,
    list_chat_sessions,
    message_event_indexes,
    rename_chat,
)
from story_rp_engine.core.agent_utils import format_sse_stream, stream_group_turn
from story_rp_engine.core.types import GroupCard, GroupChatRequest
from story_rp_engine.group.selector_agent import SPEAKER_SELECTOR
from story_rp_engine.group.workflow import group_speakers
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Group Chat"])

GROUP_APP = "group_app"


@router.post("/groups")
async def save_group(group: GroupCard, request: Request):
    try:
        await request.app.state.store.save_group(group.group_id, group)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "group_id": group.group_id}


@router.get("/groups")
async def list_groups(request: Request):
    return await request.app.state.store.list_groups()


@router.get("/groups/{group_id}")
async def get_group(group_id: str, request: Request):
    try:
        group = await request.app.state.store.get_group(group_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return group


@router.delete("/groups/{group_id}")
async def delete_group(group_id: str, request: Request):
    try:
        clean_id = _sanitize_key(group_id)
        deleted = await request.app.state.store.delete_group(clean_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Group not found")
    request.app.state.agent_registry.forget_group(clean_id)
    return {"status": "deleted", "group_id": clean_id}


@router.post("/group/chat/stream")
async def chat_group_stream(req: GroupChatRequest, request: Request):
    try:
        session_id = _sanitize_key(req.session_id)
        runner = await request.app.state.agent_registry.get_or_create_group_runner(req.group_id)
        state_delta = await chat_state_delta(req, request, group_id=req.group_id)
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e) else 400, detail=str(e))
    group = await request.app.state.store.get_group(req.group_id)

    generator = stream_group_turn(
        runner,
        user_id=USER_ID,
        session_id=session_id,
        message=req.message,
        speakers=group_speakers(group),
        state_delta=state_delta,
    )
    return StreamingResponse(
        format_sse_stream(generator, chunk_size=req.chunk_size),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/group/sessions")
async def list_group_sessions(group_id: str, request: Request):
    """Chats with one group, newest first."""
    return await list_chat_sessions(request.app.state.session_service, GROUP_APP, "group_id", group_id)


@router.get("/group/sessions/{session_id}/turns")
async def get_group_session_turns(session_id: str, request: Request):
    """The chat as the UI shows it, without the speaker selector's events. Each turn names its speaker's char_id
    (None for the user; the raw agent name for a member who has since left the group)."""
    session = await request.app.state.session_service.get_session(
        app_name=GROUP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    if not session:
        return {"turns": []}
    group = await request.app.state.store.get_group(session.state["group_id"])
    speakers = group_speakers(group) if group else {}
    turns = []
    for pos, idx in enumerate(message_event_indexes(session.events, skip_authors={SPEAKER_SELECTOR})):
        ev = session.events[idx]
        turns.append({
            "index": pos,
            "role": ev.content.role,
            "speaker": None if ev.author == "user" else speakers.get(ev.author, ev.author),
            "text": "".join(p.text for p in ev.content.parts if p.text),
        })
    return {"turns": turns}


@router.patch("/group/sessions/{session_id}")
async def rename_group_session(session_id: str, req: RenameSessionRequest, request: Request):
    await rename_chat(request.app.state.session_service, GROUP_APP, session_id, req.title)
    return {"status": "renamed", "session_id": session_id}


@router.delete("/group/sessions/{session_id}")
async def delete_group_session(session_id: str, request: Request):
    await request.app.state.session_service.delete_session(
        app_name=GROUP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    return {"status": "deleted", "session_id": session_id}


@router.post("/group/sessions/{session_id}/turns/delete")
async def delete_group_session_turn(session_id: str, req: DeleteTurnRequest, request: Request):
    remaining = await delete_chat_turn(
        request.app.state.session_service,
        GROUP_APP,
        checked_session_id(session_id),
        req.turn_index,
        req.truncate_subsequent,
        skip_authors={SPEAKER_SELECTOR},
    )
    return {"status": "ok", "remaining_turns": remaining}
```

- [ ] **Step 4: Register the router.** In `src/story_rp_engine/api/app.py`, add `from story_rp_engine.api.routes_group import router as group_router` next to the other router imports, and `app.include_router(group_router)` after `app.include_router(lorebook_router)`.

- [ ] **Step 5: Run the tests to verify they pass.**

Run: `uv run pytest tests/story_rp_engine/test_group_api.py -v`
Expected: 7 passed.

- [ ] **Step 6: Update `CLAUDE.md`** (section `story_rp_engine`). Make these edits:

Replace the sentence `plus one story workflow runner (\`story_app\`). An agent loaded from the store is rebuilt when the stored card changes, because serverless instances can share the store. Both apps get the same \`EventsCompactionConfig\`.` with:

```markdown
plus one story workflow runner (`story_app`) and one group chat runner per `group_id` (`group_app`). An RP agent or group runner loaded from the store is rebuilt when the stored card (or group, or a member's card) changes, because serverless instances can share the store. `rp_app` uses `EventsCompactionConfig` with a token threshold; the workflow apps use turn-count compaction only (see below).
```

After the `- Story (\`story/\`): ...` bullet, add:

```markdown
- Group chat (`group/`): a saved `GroupCard` (`group_id`, `name`, `char_ids`, `scenario`, `first_mes`, `lorebook_id`) puts several characters in one scene. `create_group_workflow` builds a dynamic `Workflow` per group whose `group_turn` step runs the `speaker_selector` `LlmAgent` (its `output_schema` is `{"speakers": List[Literal[<member ids>]]}`), then `ctx.run_node`s each chosen character in order; `pick_speakers` drops repeats and falls back to group order. Characters run on the branch `group` and the selector on a sub-branch of it, so the selector sees every line but characters never see its JSON (a node with no branch would see everything). Character agents are named `group_agent_name(char_id)` (`char_` + the ID with non-word characters replaced, so CJK IDs stay distinct), use `build_group_system_instruction` (the group's scenario replaces the card's) and reuse `rp_before_model_callback`. `group_app` uses turn-count compaction with `group_summary_prompt(cards)`.
```

Append to the `- Turns run through ...` bullet:

```markdown
 Group chat streams `(char_id, text)` pairs from `stream_group_turn`; `format_sse_stream` then adds `"speaker"` to each delta and ends with `{"replies": [{"speaker", "text"}], "done": true}` instead of `full_text`.
```

Replace `Every session uses \`user_id="User"\`, and the session routes hardcode the app name \`rp_app\`.` with:

```markdown
Every session uses `user_id="User"`. The RP and group session routes share the helpers in `api/chat_sessions.py`, each passing its app name (`rp_app`, `group_app`).
```

In the Storage bullet, replace `the \`story_rp_characters\`/\`story_rp_lorebooks\` tables` with `` the `story_rp_characters`/`story_rp_lorebooks`/`story_rp_groups` tables ``, and `characters and lorebooks to JSON files` with `characters, lorebooks and groups to JSON files`.

- [ ] **Step 7: Update the character-rp skill.** In `.agents/skills/character-rp/SKILL.md`, insert this section before `## Critical Guardrails` (keep the `---` separators):

```markdown
## Group Chat

A saved `GroupCard` (`group_id`, `name`, `char_ids`, `scenario`, `first_mes`, `lorebook_id`) puts several characters in one scene; the code lives in `story_rp_engine/group/`.
- Each turn the `speaker_selector` agent returns `{"speakers": [char_id, ...]}`, then each chosen character's agent replies in order (`group/workflow.py`).
- Character agents use `build_group_system_instruction`: the group's `scenario` replaces the card's, and the other members are introduced by name. They reuse `rp_before_model_callback` for lore and the Author's Note.
- Agents are named `group_agent_name(char_id)`; another member's line reaches a character as `[char_<id>] said: ...`.
- Test with `uv run pytest tests/story_rp_engine/test_group_prompt_builder.py tests/story_rp_engine/test_group_workflow.py tests/story_rp_engine/test_group_api.py -v`.

---

```

- [ ] **Step 8: Run the full suite.**

Run: `uv run pytest -q`
Expected: all pass, Postgres tests skipped. That includes `tests/test_agent_skills.py`.

- [ ] **Step 9: Commit.**

```bash
git add src/story_rp_engine/api/routes_group.py src/story_rp_engine/api/app.py tests/story_rp_engine/test_group_api.py CLAUDE.md .agents/skills/character-rp/SKILL.md
git commit -m "add group chat api"
```

---

### Task 8: Groups editor tab

**Files:**
- Modify: `src/story_rp_engine/web/app.js`
- Modify: `src/story_rp_engine/web/index.html`
- Test: `tests/test_frontend_integration.py`

**Interfaces:**
- Consumes: `GET/POST /api/v1/groups`, `DELETE /api/v1/groups/{id}`.
- Produces (used by Task 9): `this.groups` (`{group_id: GroupCard}`), `loadGroups()`, and `characterName(charId) -> string`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_frontend_integration.py`:

```python
def test_groups_tab_is_served(client):
    html = client.get("/").text
    assert "activeTab === 'groups'" in html
    assert "saveGroup" in html
    js = client.get("/app.js").text
    assert "'/api/v1/groups'" in js
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `uv run pytest tests/test_frontend_integration.py -k groups_tab -v`
Expected: FAIL on `assert "activeTab === 'groups'" in html`.

- [ ] **Step 3: Add the state, computed property and methods to `app.js`.**

Change the `activeTab` comment to `// 'roleplay' | 'group' | 'story' | 'characters' | 'groups' | 'lorebooks'`.

In `data()`, after the Lorebooks Management State block (after `lorebookForm: {...},`), add:

```js
      // =======================================================================
      // Groups Management State
      // =======================================================================
      groups: {}, // Object mapping group_id -> GroupCard
      selectedGroupId: null,
      groupSearchQuery: '',
      groupForm: { group_id: '', name: '', char_ids: [], scenario: '', first_mes: '', lorebook_id: '' },
```

In `computed`, after `filteredLorebooks() {...},`, add:

```js
    filteredGroups() {
      const query = (this.groupSearchQuery || '').toLowerCase().trim();
      const list = Object.values(this.groups || {});
      if (!query) return list;
      return list.filter((g) =>
        [g.name, g.group_id, g.scenario].some((s) => (s || '').toLowerCase().includes(query)),
      );
    },
```

In `methods`, before the `// Roleplay Tab Reactive Logic` banner, add:

```js
    // =========================================================================
    // Groups Management
    // =========================================================================
    async loadGroups() {
      try {
        const res = await fetch('/api/v1/groups');
        if (res.ok) this.groups = await res.json();
      } catch (err) {
        console.error('Failed to load groups:', err);
      }
      this.refreshIcons();
    },

    characterName(charId) {
      const c = this.characters.find((x) => x.char_id === charId);
      return c ? c.name : charId;
    },

    selectGroup(id) {
      const g = this.groups[id];
      if (!g) return;
      this.selectedGroupId = id;
      this.groupForm = {
        group_id: g.group_id,
        name: g.name || '',
        char_ids: [...(g.char_ids || [])],
        scenario: g.scenario || '',
        first_mes: g.first_mes || '',
        lorebook_id: g.lorebook_id || '',
      };
      this.refreshIcons();
    },

    newGroup() {
      this.selectedGroupId = null;
      this.groupForm = { group_id: '', name: '', char_ids: [], scenario: '', first_mes: '', lorebook_id: '' };
      this.refreshIcons();
    },

    // Members speak in the order they were added when the speaker selector picks no one.
    toggleGroupMember(charId) {
      const ids = this.groupForm.char_ids;
      const i = ids.indexOf(charId);
      if (i >= 0) ids.splice(i, 1);
      else ids.push(charId);
    },

    async saveGroup() {
      const groupId = (this.groupForm.group_id || '').trim();
      const name = (this.groupForm.name || '').trim();
      if (!groupId || !name) {
        this.showToast('Group ID and name are required.', 'error');
        return;
      }
      if (this.groupForm.char_ids.length === 0) {
        this.showToast('Add at least one character to the group.', 'error');
        return;
      }
      const payload = {
        group_id: groupId,
        name: name,
        char_ids: [...this.groupForm.char_ids],
        scenario: this.groupForm.scenario || '',
        first_mes: this.groupForm.first_mes || '',
        lorebook_id: this.groupForm.lorebook_id || null,
      };
      try {
        const res = await fetch('/api/v1/groups', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          this.showToast(`Group "${name}" saved!`, 'success');
          this.selectedGroupId = groupId;
          await this.loadGroups();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to save group' }));
          this.showToast(err.detail || 'Save failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while saving group', 'error');
      }
    },

    async deleteGroup(id) {
      if (!id || !confirm(`Are you sure you want to delete group "${id}"?`)) return;
      try {
        const res = await fetch(`/api/v1/groups/${encodeURIComponent(id)}`, { method: 'DELETE' });
        if (res.ok) {
          this.showToast(`Group "${id}" deleted.`, 'success');
          if (this.selectedGroupId === id) this.newGroup();
          await this.loadGroups();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to delete' }));
          this.showToast(err.detail || 'Delete failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while deleting group', 'error');
      }
    },
```

In `mounted()`, add `this.loadGroups();` after `this.loadLorebooks();`.

- [ ] **Step 4: Add the nav button and the tab to `index.html`.**

In `<nav>`, after the Characters button, add:

```html
        <button
          @click="activeTab = 'groups'"
          :class="activeTab === 'groups' ? 'bg-indigo-600 text-white font-medium shadow-sm' : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'"
          class="flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs transition">
          <i data-lucide="contact" class="w-3.5 h-3.5"></i>
          <span>Groups</span>
        </button>
```

Immediately before the `<!-- TAB 4: LOREBOOKS MANAGEMENT WORKBENCH -->` comment block (before its opening `<!-- ===... -->` line), add:

```html
      <!-- ================================================================= -->
      <!-- GROUPS MANAGEMENT WORKBENCH                                       -->
      <!-- ================================================================= -->
      <section v-show="activeTab === 'groups'" class="w-full h-full flex flex-col md:flex-row overflow-hidden">

        <!-- Left Column: Group Directory -->
        <aside class="w-full md:w-80 lg:w-96 flex-shrink-0 border-r border-neutral-800 bg-neutral-900/60 overflow-y-auto p-4 space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-neutral-800">
            <h2 class="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center space-x-1.5">
              <i data-lucide="contact" class="w-3.5 h-3.5 text-indigo-400"></i>
              <span>Groups Directory</span>
            </h2>
            <span class="text-[10px] font-mono text-neutral-500">{{ filteredGroups.length }} saved</span>
          </div>

          <button
            @click="newGroup"
            class="w-full flex items-center justify-center space-x-1 px-2.5 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition shadow-sm">
            <i data-lucide="plus" class="w-3 h-3"></i>
            <span>New Group</span>
          </button>

          <div class="relative">
            <input
              type="text"
              v-model="groupSearchQuery"
              placeholder="Search groups..."
              class="w-full bg-neutral-950 border border-neutral-800 rounded-md pl-8 pr-3 py-1.5 text-xs text-neutral-200 placeholder-neutral-500 focus:border-indigo-500">
            <i data-lucide="search" class="w-3.5 h-3.5 absolute left-2.5 top-2 text-neutral-500"></i>
          </div>

          <div class="space-y-2">
            <div v-if="filteredGroups.length === 0" class="text-center py-6 text-xs text-neutral-500">
              No groups found.
            </div>
            <div
              v-for="g in filteredGroups"
              :key="g.group_id"
              @click="selectGroup(g.group_id)"
              :class="selectedGroupId === g.group_id ? 'border-indigo-500 bg-indigo-950/20' : 'border-neutral-800 bg-neutral-950/50 hover:border-neutral-700'"
              class="p-3 rounded-lg border cursor-pointer transition space-y-1.5 select-text">
              <div class="flex items-center justify-between">
                <span class="font-medium text-xs text-neutral-200">{{ g.name || g.group_id }}</span>
                <span class="px-1.5 py-0.5 rounded bg-neutral-800 text-[10px] font-mono text-indigo-400">
                  {{ g.char_ids.length }} members
                </span>
              </div>
              <p class="text-[11px] text-neutral-400 line-clamp-2 leading-relaxed">
                {{ g.char_ids.map(characterName).join(', ') }}
              </p>
            </div>
          </div>
        </aside>

        <!-- Right Column: Group Editor -->
        <div class="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 bg-neutral-950">
          <div class="flex items-center justify-between pb-4 border-b border-neutral-800">
            <div>
              <h2 class="text-sm md:text-base font-bold text-neutral-100 flex items-center space-x-2">
                <span>{{ groupForm.name || 'New Group' }}</span>
                <span v-if="selectedGroupId" class="text-xs font-mono text-neutral-500">({{ selectedGroupId }})</span>
              </h2>
              <p class="text-xs text-neutral-400">Characters who share one scene in Group Chat.</p>
            </div>
            <div class="flex items-center space-x-2">
              <button
                v-if="selectedGroupId"
                @click="deleteGroup(selectedGroupId)"
                class="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-rose-600/80 hover:bg-rose-600 text-white text-xs font-medium transition shadow-sm">
                <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                <span>Delete</span>
              </button>
              <button
                @click="saveGroup"
                class="flex items-center space-x-1.5 px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition shadow-sm">
                <i data-lucide="save" class="w-3.5 h-3.5"></i>
                <span>Save Group</span>
              </button>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="space-y-1.5">
              <label class="block text-xs font-medium text-neutral-300">Group ID (Slug) <span class="text-rose-400">*</span></label>
              <input
                type="text"
                v-model="groupForm.group_id"
                placeholder="e.g. tavern_regulars"
                class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 font-mono focus:border-indigo-500">
            </div>
            <div class="space-y-1.5">
              <label class="block text-xs font-medium text-neutral-300">Name <span class="text-rose-400">*</span></label>
              <input
                type="text"
                v-model="groupForm.name"
                placeholder="e.g. Tavern Regulars"
                class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
            </div>
          </div>

          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">Members <span class="text-rose-400">*</span></label>
            <p class="text-[11px] text-neutral-500">Click to add or remove. The number is the speaking order used when the speaker selector picks no one.</p>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              <button
                v-for="c in characters"
                :key="c.char_id"
                @click="toggleGroupMember(c.char_id)"
                :class="groupForm.char_ids.includes(c.char_id) ? 'border-indigo-500 bg-indigo-950/30 text-neutral-100' : 'border-neutral-800 bg-neutral-900 text-neutral-400 hover:border-neutral-700'"
                class="flex items-center justify-between px-3 py-1.5 rounded-md border text-xs text-left transition">
                <span class="truncate">{{ c.name }} <span class="font-mono text-neutral-500">({{ c.char_id }})</span></span>
                <span
                  v-if="groupForm.char_ids.includes(c.char_id)"
                  class="ml-2 px-1.5 rounded bg-indigo-600 text-white text-[10px] font-mono">
                  {{ groupForm.char_ids.indexOf(c.char_id) + 1 }}
                </span>
              </button>
            </div>
            <p v-if="characters.length === 0" class="text-[11px] text-neutral-500">No characters yet. Create some in the Characters tab.</p>
          </div>

          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">Scenario</label>
            <textarea
              v-model="groupForm.scenario"
              rows="4"
              placeholder="The shared scene. Replaces each member's own scenario in group chats."
              class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-2 text-xs text-neutral-200 placeholder-neutral-600 focus:border-indigo-500"></textarea>
          </div>

          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">Opening Message</label>
            <textarea
              v-model="groupForm.first_mes"
              rows="4"
              placeholder="Shown before the user's first message, e.g. narration that sets the scene."
              class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-2 text-xs text-neutral-200 placeholder-neutral-600 focus:border-indigo-500"></textarea>
          </div>

          <div class="space-y-1.5 pb-8">
            <label class="block text-xs font-medium text-neutral-300">Default Lorebook</label>
            <select
              v-model="groupForm.lorebook_id"
              class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
              <option value="">None (No Lorebook)</option>
              <option v-for="(lb, id) in lorebooks" :key="id" :value="id">
                {{ lb.name || id }}
              </option>
            </select>
          </div>
        </div>

      </section>

```

- [ ] **Step 5: Run the frontend tests.**

Run: `uv run pytest tests/test_frontend_integration.py tests/test_web_mount.py -v`
Expected: all pass.

- [ ] **Step 6: Commit.**

```bash
git add src/story_rp_engine/web/app.js src/story_rp_engine/web/index.html tests/test_frontend_integration.py
git commit -m "add groups editor tab"
```

---

### Task 9: Group Chat tab

**Files:**
- Modify: `src/story_rp_engine/web/app.js`
- Modify: `src/story_rp_engine/web/index.html`
- Test: `tests/test_frontend_integration.py`

**Interfaces:**
- Consumes: `this.groups`, `loadGroups()`, `characterName()` (Task 8); the group chat and session endpoints (Task 7). The existing `startRename`, `cancelRename`, `saveRename`, `formatSessionTime`, `copyMessage`, `renderMarkdown`, `showToast` and `refreshIcons`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_frontend_integration.py`:

```python
def test_group_chat_tab_is_served(client):
    html = client.get("/").text
    assert "activeTab === 'group'" in html
    assert "group-chat-feed" in html
    js = client.get("/app.js").text
    assert "/api/v1/group/chat/stream" in js
    assert "/api/v1/group/sessions" in js
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `uv run pytest tests/test_frontend_integration.py -k group_chat_tab -v`
Expected: FAIL on `assert "activeTab === 'group'" in html`.

- [ ] **Step 3: Add the state and computed properties to `app.js`.**

In `data()`, after the Roleplay Tab state block (after `editingTitle: '',`), add:

```js
      // =======================================================================
      // Group Chat Reactive State
      // =======================================================================
      groupChatId: '', // group being chatted with
      groupSessionId: 'group_' + Math.random().toString(36).substring(2, 10),
      groupUserName: 'User',
      groupAuthorsNote: '',
      groupLorebookId: '',
      groupLorebookSentKey: null, // "<session>|<lorebook>" last loaded into the backend session
      groupChunkSize: 1,
      groupMessages: [], // { role: 'user' | 'assistant', speaker: char_id | null, content, timestamp, isGreeting? }
      isGeneratingGroup: false,
      groupAbortController: null,
      groupInput: '',
      groupSessions: [], // past chats with the selected group, newest first
      showGroupHistory: false,
```

In `computed`, after `storySessionTitle() {...},`, add:

```js
    groupSessionTitle() {
      const s = this.groupSessions.find((x) => x.session_id === this.groupSessionId);
      return (s && (s.title || s.last_message)) || 'New chat';
    },
    activeGroup() {
      return this.groups[this.groupChatId] || null;
    },
```

- [ ] **Step 4: Let `saveRename` handle group chats.** In `saveRename`, replace

```js
      const base = kind === 'rp' ? '/api/v1/rp/sessions' : '/api/v1/story/sessions';
```

with

```js
      const base = {
        rp: '/api/v1/rp/sessions',
        group: '/api/v1/group/sessions',
        story: '/api/v1/story/sessions',
      }[kind];
```

and update the comment above it to `// kind is 'rp', 'group' or 'story'. Runs on Enter and on blur; the first call wins.`

- [ ] **Step 5: Add the group chat methods.** In `methods`, before the `// Story Co-Pilot Chat` banner, add:

```js
    // =========================================================================
    // Group Chat: the speaker selector picks who replies; each reply streams into its own bubble
    // =========================================================================
    timeNow() {
      return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    scrollGroupChatToBottom() {
      this.$nextTick(() => {
        const feed = document.getElementById('group-chat-feed');
        if (feed) feed.scrollTop = feed.scrollHeight;
      });
    },

    groupSpeakerName(msg) {
      if (msg.role === 'user') return this.groupUserName || 'User';
      if (msg.isGreeting) return this.activeGroup ? this.activeGroup.name : 'Narrator';
      return this.characterName(msg.speaker);
    },

    onGroupChatChange() {
      // Each group gets its own chats, so switching starts a new session.
      this.newGroupSession();
      this.loadGroupSessions();
    },

    newGroupSession() {
      this.stopGeneratingGroup();
      this.groupSessionId = 'group_' + Math.random().toString(36).substring(2, 10);
      this.groupMessages = [];
      this.groupLorebookSentKey = null;
      const group = this.activeGroup;
      this.groupLorebookId = (group && group.lorebook_id) || '';
      if (group && group.first_mes) {
        this.groupMessages.push({
          role: 'assistant',
          speaker: null,
          content: group.first_mes,
          isGreeting: true,
          timestamp: this.timeNow(),
        });
      }
      this.refreshIcons();
      this.scrollGroupChatToBottom();
    },

    toggleGroupHistory() {
      this.showGroupHistory = !this.showGroupHistory;
      if (this.showGroupHistory) this.loadGroupSessions();
    },

    async loadGroupSessions() {
      if (!this.groupChatId) {
        this.groupSessions = [];
        return;
      }
      try {
        const res = await fetch(`/api/v1/group/sessions?group_id=${encodeURIComponent(this.groupChatId)}`);
        if (res.ok) this.groupSessions = await res.json();
      } catch (err) {
        console.warn('Failed to load group chat history:', err);
      }
      this.refreshIcons();
    },

    async openGroupSession(s) {
      this.stopGeneratingGroup();
      try {
        const res = await fetch(`/api/v1/group/sessions/${encodeURIComponent(s.session_id)}/turns`);
        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
        const { turns } = await res.json();
        const messages = turns.map((t) => ({
          role: t.role === 'user' ? 'user' : 'assistant',
          speaker: t.speaker,
          content: t.text,
          timestamp: '',
        }));
        if (s.greeting) {
          messages.unshift({ role: 'assistant', speaker: null, content: s.greeting, isGreeting: true, timestamp: '' });
        }
        this.groupSessionId = s.session_id;
        this.groupMessages = messages;
        this.groupUserName = s.user_name || 'User';
        this.groupAuthorsNote = s.authors_note || '';
        this.groupLorebookId = s.lorebook_id || '';
        this.groupLorebookSentKey = null; // re-send the selected lorebook with the next message
        this.showGroupHistory = false;
        this.refreshIcons();
        this.scrollGroupChatToBottom();
      } catch (err) {
        this.showToast(err.message || 'Failed to open chat.', 'error');
      }
    },

    async deleteGroupSession(s) {
      if (!confirm('Delete this chat? This cannot be undone.')) return;
      await fetch(`/api/v1/group/sessions/${encodeURIComponent(s.session_id)}`, { method: 'DELETE' });
      if (s.session_id === this.groupSessionId) this.newGroupSession();
      this.loadGroupSessions();
    },

    stopGeneratingGroup() {
      if (this.groupAbortController) this.groupAbortController.abort();
      this.groupAbortController = null;
      this.isGeneratingGroup = false;
    },

    async deleteGroupTurn(index) {
      const msg = this.groupMessages[index];
      if (!msg || this.isGeneratingGroup) return;
      // The greeting lives only in the UI; the backend counts the messages after it.
      if (!msg.isGreeting) {
        const hasGreeting = !!this.groupMessages[0].isGreeting;
        try {
          const res = await fetch(`/api/v1/group/sessions/${encodeURIComponent(this.groupSessionId)}/turns/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ turn_index: hasGreeting ? index - 1 : index, truncate_subsequent: false }),
          });
          if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
        } catch (err) {
          this.showToast(err.message || 'Failed to delete message.', 'error');
          return;
        }
      }
      this.groupMessages.splice(index, 1);
    },

    async sendGroupMessage() {
      const text = (this.groupInput || '').trim();
      if (this.isGeneratingGroup || !text) return;
      if (!this.groupChatId) {
        this.showToast('Please select a group first.', 'error');
        return;
      }
      this.groupInput = '';
      this.groupMessages.push({ role: 'user', speaker: null, content: text, timestamp: this.timeNow() });
      this.scrollGroupChatToBottom();
      await this._streamGroupReplies(text);
    },

    async _streamGroupReplies(promptText) {
      this.isGeneratingGroup = true;
      this.groupAbortController = new AbortController();
      this.refreshIcons();

      const firstMsg = this.groupMessages[0];
      const payload = {
        group_id: this.groupChatId,
        session_id: this.groupSessionId,
        message: promptText,
        // The greeting lives only in the UI, so the backend gets it with each message.
        greeting: firstMsg && firstMsg.isGreeting ? firstMsg.content : null,
        authors_note: this.groupAuthorsNote ? this.groupAuthorsNote.trim() : null,
        user_name: this.groupUserName ? this.groupUserName.trim() : 'User',
        chunk_size: Number(this.groupChunkSize) || 1,
      };
      // Like RP: the backend keeps the lorebook in the session, so only send it when it changed ('' clears it).
      const lorebookKey = `${this.groupSessionId}|${this.groupLorebookId || ''}`;
      if (lorebookKey !== this.groupLorebookSentKey) {
        payload.lorebook_id = this.groupLorebookId || '';
      }

      let current = null; // the bubble being streamed; a new speaker opens a new one
      const handle = (data) => {
        if (data.error) throw new Error(data.error);
        if (!data.delta) return;
        if (!current || current.speaker !== data.speaker) {
          this.groupMessages.push({ role: 'assistant', speaker: data.speaker, content: '', timestamp: this.timeNow() });
          // Mutate the reactive copy in the array so Vue re-renders as text arrives.
          current = this.groupMessages[this.groupMessages.length - 1];
        }
        current.content += data.delta;
        this.scrollGroupChatToBottom();
      };

      try {
        const res = await fetch('/api/v1/group/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: this.groupAbortController.signal,
        });
        if (!res.ok) {
          let detail = `Server returned HTTP ${res.status}`;
          try {
            const errJson = await res.json();
            if (errJson.detail) detail = errJson.detail;
          } catch (_) {}
          throw new Error(detail);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
          const blocks = buffer.split('\n\n');
          buffer = blocks.pop();
          for (const block of blocks) {
            const line = block.trim();
            if (!line.startsWith('data:')) continue;
            const raw = line.slice(5).trim();
            if (raw === '[DONE]') continue;
            let data;
            try {
              data = JSON.parse(raw);
            } catch (_) {
              continue; // ignore a malformed chunk
            }
            handle(data);
          }
        }

        // Marked only after a completed turn; re-sending on a failed one is harmless.
        this.groupLorebookSentKey = lorebookKey;
        if (!current) this.showToast('No character replied.', 'info');
      } catch (err) {
        if (err.name === 'AbortError') {
          this.showToast('Generation stopped.', 'info');
        } else {
          console.error('Group chat streaming error:', err);
          this.showToast(err.message || 'Error generating group replies.', 'error');
        }
      } finally {
        this.isGeneratingGroup = false;
        this.groupAbortController = null;
        this.loadGroupSessions();
        this.scrollGroupChatToBottom();
        this.refreshIcons();
      }
    },
```

- [ ] **Step 6: Add the nav button and the tab to `index.html`.**

In `<nav>`, after the Roleplay button, add:

```html
        <button
          @click="activeTab = 'group'"
          :class="activeTab === 'group' ? 'bg-indigo-600 text-white font-medium shadow-sm' : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'"
          class="flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs transition">
          <i data-lucide="messages-square" class="w-3.5 h-3.5"></i>
          <span>Group Chat</span>
        </button>
```

Immediately before the `<!-- TAB 2: STORY CO-PILOT WORKBENCH -->` comment block (before its opening `<!-- ===... -->` line), add:

```html
      <!-- ================================================================= -->
      <!-- GROUP CHAT WORKBENCH                                              -->
      <!-- ================================================================= -->
      <section v-show="activeTab === 'group'" class="w-full h-full flex flex-col md:flex-row overflow-hidden">

        <!-- Left Column: Group Chat Setup -->
        <aside class="w-full md:w-80 lg:w-96 flex-shrink-0 border-r border-neutral-800 bg-neutral-900/60 overflow-y-auto p-4 space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-neutral-800">
            <h2 class="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center space-x-1.5">
              <i data-lucide="sliders" class="w-3.5 h-3.5 text-indigo-400"></i>
              <span>Group Chat Setup</span>
            </h2>
          </div>

          <!-- Group Selector -->
          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">Active Group</label>
            <select
              v-model="groupChatId"
              @change="onGroupChatChange"
              class="w-full bg-neutral-950 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
              <option value="" disabled>-- Select Group --</option>
              <option v-for="(g, id) in groups" :key="id" :value="id">
                {{ g.name }} ({{ g.char_ids.length }} members)
              </option>
            </select>
            <p v-if="activeGroup" class="text-[11px] text-neutral-500">{{ activeGroup.char_ids.map(characterName).join(', ') }}</p>
          </div>

          <!-- Session Controls -->
          <div class="space-y-1.5">
            <div class="flex items-center justify-between">
              <label class="block text-xs font-medium text-neutral-300">Chat</label>
              <button
                @click="newGroupSession"
                :disabled="!groupChatId"
                class="text-[11px] text-indigo-400 hover:text-indigo-300 hover:underline disabled:opacity-40">
                New
              </button>
            </div>
            <div class="relative">
              <button
                @click="toggleGroupHistory"
                :disabled="!groupChatId"
                class="w-full flex items-center justify-between gap-2 bg-neutral-950 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 hover:border-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Switch, rename or delete chats">
                <span class="truncate">{{ groupSessionTitle }}</span>
                <i data-lucide="chevron-down" class="w-3.5 h-3.5 text-neutral-500 flex-shrink-0"></i>
              </button>
              <div v-if="showGroupHistory" class="fixed inset-0 z-10" @click="showGroupHistory = false"></div>
              <div
                v-if="showGroupHistory"
                class="absolute left-0 right-0 mt-1 max-h-72 overflow-y-auto z-20 bg-neutral-900 border border-neutral-800 rounded-md shadow-lg p-1 space-y-1">
                <p v-if="groupSessions.length === 0" class="px-2.5 py-2 text-[11px] text-neutral-500">No saved chats with this group yet.</p>
                <div
                  v-for="s in groupSessions"
                  :key="s.session_id"
                  @click="editingSessionId !== s.session_id && openGroupSession(s)"
                  :class="s.session_id === groupSessionId ? 'bg-indigo-950/40' : 'hover:bg-neutral-800/60'"
                  class="group flex items-start justify-between gap-2 px-2.5 py-1.5 rounded cursor-pointer">
                  <div class="min-w-0 flex-1">
                    <input
                      v-if="editingSessionId === s.session_id"
                      :id="`title-input-${s.session_id}`"
                      v-model="editingTitle"
                      @click.stop
                      @keydown.enter.prevent="saveRename('group', s)"
                      @keydown.esc.prevent="cancelRename"
                      @blur="saveRename('group', s)"
                      placeholder="Title"
                      class="w-full bg-neutral-950 border border-indigo-600 rounded px-1.5 py-0.5 text-xs text-neutral-200">
                    <p v-else class="text-xs text-neutral-200 truncate">{{ s.title || s.last_message || s.session_id }}</p>
                    <p class="text-[10px] text-neutral-500">{{ formatSessionTime(s.updated_at) }}</p>
                  </div>
                  <div class="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 flex-shrink-0">
                    <button @click.stop="startRename(s)" class="text-neutral-500 hover:text-indigo-300" title="Rename">
                      <i data-lucide="pencil" class="w-3.5 h-3.5"></i>
                    </button>
                    <button @click.stop="deleteGroupSession(s)" class="text-neutral-500 hover:text-rose-400" title="Delete">
                      <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- User Name / Persona -->
          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">User Persona Name</label>
            <input
              type="text"
              v-model="groupUserName"
              placeholder="User"
              class="w-full bg-neutral-950 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
          </div>

          <!-- Active Lorebook -->
          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">Active Lorebook</label>
            <select
              v-model="groupLorebookId"
              class="w-full bg-neutral-950 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
              <option value="">None (No Lorebook)</option>
              <option v-for="(lb, id) in lorebooks" :key="id" :value="id">
                {{ lb.name || id }}
              </option>
            </select>
          </div>

          <!-- Author's Note -->
          <div class="space-y-1.5">
            <div class="flex items-center justify-between">
              <label class="block text-xs font-medium text-neutral-300">Author's Note</label>
              <span class="text-[10px] text-neutral-500 font-mono">Injected Prompt Steering</span>
            </div>
            <textarea
              v-model="groupAuthorsNote"
              rows="3"
              placeholder="[Pacing: slow; Focus on banter between the characters]"
              class="w-full bg-neutral-950 border border-neutral-800 rounded-md px-3 py-2 text-xs text-neutral-200 placeholder-neutral-600 focus:border-indigo-500"></textarea>
          </div>

          <!-- Chunk Size Slider -->
          <div class="space-y-1.5 pt-2 border-t border-neutral-800">
            <div class="flex items-center justify-between text-xs">
              <span class="text-neutral-300">SSE Chunk Size</span>
              <span class="font-mono text-indigo-400 font-semibold">{{ groupChunkSize }} {{ groupChunkSize === 1 ? 'token (Instant)' : 'tokens' }}</span>
            </div>
            <input
              type="range"
              min="1"
              max="64"
              v-model.number="groupChunkSize"
              class="w-full accent-indigo-500 h-1.5 bg-neutral-800 rounded-lg cursor-pointer">
          </div>
        </aside>

        <!-- Right Column: Group Chat Feed -->
        <div class="flex-1 flex flex-col h-full overflow-hidden bg-neutral-950">

          <div class="flex-1 overflow-y-auto p-4 md:p-6 space-y-4" id="group-chat-feed">
            <div v-if="groupMessages.length === 0" class="h-full flex flex-col items-center justify-center text-center p-8 text-neutral-500">
              <i data-lucide="messages-square" class="w-12 h-12 text-neutral-700 mb-3"></i>
              <h3 class="text-sm font-semibold text-neutral-400 mb-1">No Messages Yet</h3>
              <p class="text-xs max-w-sm text-neutral-500">
                Select a group on the left and send a message. The characters decide who answers.
              </p>
            </div>

            <div
              v-for="(msg, idx) in groupMessages"
              :key="idx"
              :class="msg.role === 'user' ? 'items-end' : 'items-start'"
              class="flex flex-col group space-y-1">
              <div
                :class="msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : 'flex-row'"
                class="flex items-center space-x-2 text-[11px] text-neutral-400 px-1">
                <span class="font-semibold text-neutral-300">{{ groupSpeakerName(msg) }}</span>
                <span class="text-neutral-600">&bull;</span>
                <span class="text-neutral-500 text-[10px]">{{ msg.timestamp || 'now' }}</span>
                <div class="opacity-0 group-hover:opacity-100 transition-opacity flex items-center space-x-1 bg-neutral-900 border border-neutral-800 rounded px-1 py-0.5 ml-2">
                  <button
                    @click="copyMessage(msg.content)"
                    title="Copy text"
                    class="p-1 hover:text-indigo-400 text-neutral-400 transition">
                    <i data-lucide="copy" class="w-3 h-3"></i>
                  </button>
                  <button
                    @click="deleteGroupTurn(idx)"
                    title="Delete message"
                    class="p-1 hover:text-rose-400 text-neutral-400 transition">
                    <i data-lucide="trash-2" class="w-3 h-3"></i>
                  </button>
                </div>
              </div>
              <div
                :class="msg.role === 'user'
                  ? 'bg-indigo-950/40 border-indigo-800/50 rounded-2xl rounded-tr-sm text-neutral-100'
                  : 'bg-neutral-900/90 border-neutral-800 rounded-2xl rounded-tl-sm text-neutral-200'"
                class="border px-4 py-3 max-w-2xl select-text shadow-sm text-sm leading-relaxed">
                <div class="prose-dark" v-html="renderMarkdown(msg.content)"></div>
                <span
                  v-if="isGeneratingGroup && idx === groupMessages.length - 1 && msg.role === 'assistant'"
                  class="cursor-blink">▌</span>
              </div>
            </div>
          </div>

          <!-- Bottom Input Area -->
          <div class="border-t border-neutral-800 bg-neutral-900/70 p-3 md:p-4 flex-shrink-0">
            <div class="flex flex-col space-y-2 max-w-4xl mx-auto">
              <textarea
                v-model="groupInput"
                @keydown.enter.exact.prevent="sendGroupMessage"
                rows="3"
                placeholder="Talk to the group... (Press Enter to send, Shift+Enter for newline)"
                class="w-full bg-neutral-950 border border-neutral-800 rounded-lg p-3 text-xs md:text-sm text-neutral-200 placeholder-neutral-500 focus:border-indigo-500 resize-none select-text"></textarea>
              <div class="flex items-center justify-end text-xs space-x-2">
                <button
                  v-if="isGeneratingGroup"
                  @click="stopGeneratingGroup"
                  class="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-rose-600 hover:bg-rose-500 text-white font-medium transition shadow-sm">
                  <i data-lucide="square" class="w-3.5 h-3.5"></i>
                  <span>Stop Generating</span>
                </button>
                <button
                  v-else
                  @click="sendGroupMessage"
                  :disabled="!groupInput || !groupInput.trim() || !groupChatId"
                  :class="(!groupInput || !groupInput.trim() || !groupChatId) ? 'opacity-40 cursor-not-allowed bg-neutral-800 text-neutral-400' : 'bg-indigo-600 hover:bg-indigo-500 text-white'"
                  class="flex items-center space-x-1.5 px-4 py-1.5 rounded-md font-medium transition shadow-sm">
                  <i data-lucide="send" class="w-3.5 h-3.5"></i>
                  <span>Send</span>
                </button>
              </div>
            </div>
          </div>

        </div>

      </section>

```

- [ ] **Step 7: Run the frontend tests and the full suite.**

Run: `uv run pytest -q`
Expected: all pass, Postgres tests skipped.

- [ ] **Step 8: Smoke test in a browser (needs a model configured in `.env`).** Run `./run_sh/run_story_rp_backend.sh` and open http://localhost:8000. In the Groups tab, create a group of two saved characters with a scenario and an opening message. In Group Chat, select the group and send a message that addresses one character by name. Check that:
- That character answers first, in its own labelled bubble.
- The other bubbles stream in one after another.
- Reopening the chat from history shows the same speakers.
- Deleting one reply removes exactly that bubble after a reload.

If no model is configured, skip this step and say so in the hand-off.

- [ ] **Step 9: Commit.**

```bash
git add src/story_rp_engine/web/app.js src/story_rp_engine/web/index.html tests/test_frontend_integration.py
git commit -m "add group chat tab"
```
