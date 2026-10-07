# User Personas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Users save personas (name + description) in a Personas tab and pick one in the RP and group chat sidebars; the characters see the persona's name and description.

**Architecture:** `Persona` is a fourth saved entity in `EngineStore` (next to characters, lorebooks, groups) with CRUD routes. Chat requests carry `persona_id`; `chat_state_delta` loads the persona each turn and writes `persona_id`, `user_name` and `user_persona` into session state. The RP and group character agents' callable instructions read `user_persona` from state and the prompt builders add a `<user_persona>` section after `<scenario>`.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, Google ADK, SQLAlchemy async, Vue 3 (CDN, no build), pytest with `anyio`. Run everything with `uv run`.

**Spec:** `docs/superpowers/specs/2026-10-07-user-personas-design.md`

## Global Constraints

- Keep it simple (CLAUDE.md): no new helpers, config options or abstractions beyond what each task shows; no defensive checks for states our own code produces.
- Session state keys are exactly `persona_id`, `user_name`, `user_persona`.
- Missing persona on a chat turn: `HTTPException(404, "Persona not found: <id>")`. Missing persona on `GET`/`DELETE /api/v1/personas/{id}`: `404`, detail `"Persona not found"`.
- Without `persona_id`: `persona_id = None`, `user_name = req.user_name or "User"`, `user_persona = ""`.
- Prompt section format: `<user_persona>\n{description with {{user}}/{{char}} filled}\n</user_persona>`, right after `<scenario>`, only when the description is non-empty.
- The speaker selector's prompt is unchanged.
- SQL table `story_rp_personas` with columns `persona_id`, `persona_json`, `updated_at`; file storage under `<storage_dir>/personas/`.
- No new dependencies; `vercel/requirements.txt` is not touched.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A persona description with braces (`{lucky}`, `{{char}}`) must not break ADK's `{state}` templating, and `{{char}}` must fill with whichever character is speaking. Tests: Task 2 (group `{{char}}` per speaker), Task 3 (`{lucky}` through the RP API).
2. Editing a persona between two turns of an existing chat: the next turn uses the new name and description. Test: Task 3.
3. Dropping the persona mid-chat (no `persona_id`): the next turn uses `user_name` and the old description is gone from the prompt. Test: Task 3.
4. An unknown `persona_id` on a group chat returns `404` before any model call, like RP. Test: Task 3.
5. A persona ID with path traversal (`../evil`) is rejected with `400` on save. Test: Task 1.

## File Map

| File | Change |
|---|---|
| `src/story_rp_engine/core/types.py` | `Persona` model; `persona_id` on `RPChatRequest`, `GroupChatRequest` |
| `src/story_rp_engine/storage/store.py` | `story_rp_personas` table, `personas/` dir, `save/get/list/delete_persona` |
| `src/story_rp_engine/api/routes_persona.py` (new) | Persona CRUD routes |
| `src/story_rp_engine/api/app.py` | Register the persona router |
| `src/story_rp_engine/rp/prompt_builder.py` | `user_persona` parameter and section |
| `src/story_rp_engine/group/prompt_builder.py` | `user_persona` parameter and section |
| `src/story_rp_engine/api/chat_sessions.py` | Resolve `persona_id` in `chat_state_delta`; return `persona_id` from `list_chat_sessions` |
| `src/story_rp_engine/rp/agent.py`, `src/story_rp_engine/group/char_agent.py` | Pass `user_persona` from state |
| `src/story_rp_engine/web/index.html`, `src/story_rp_engine/web/app.js` | Personas tab; persona dropdowns in the chat sidebars |
| `CLAUDE.md`, `.agents/skills/story-rp-backend/**`, `.agents/skills/story-rp-frontend/**`, `.agents/skills/character-rp/references/character-card-v2-spec.md` | Docs |

---

### Task 1: Persona model, storage and CRUD routes

**Files:**
- Modify: `src/story_rp_engine/core/types.py` (after `GroupCard`)
- Modify: `src/story_rp_engine/storage/store.py`
- Create: `src/story_rp_engine/api/routes_persona.py`
- Modify: `src/story_rp_engine/api/app.py:11-14,143-146`
- Test: `tests/story_rp_engine/test_storage.py`, `tests/story_rp_engine/test_api.py`
- Docs: `CLAUDE.md`, `.agents/skills/story-rp-backend/SKILL.md`, `.agents/skills/story-rp-backend/references/database-session-storage.md`, `.agents/skills/story-rp-backend/references/api-endpoints-and-sse.md`

**Interfaces:**
- Produces: `Persona(persona_id: str, name: str, description: str = "")` in `story_rp_engine.core.types`; `EngineStore.save_persona(persona_id: str, persona: Persona) -> None`, `get_persona(persona_id: str) -> Optional[Persona]` (raises `ValueError` on an invalid ID), `list_personas() -> Dict[str, Persona]`, `delete_persona(persona_id: str) -> bool`; routes `POST/GET /api/v1/personas`, `GET/DELETE /api/v1/personas/{persona_id}`.

- [ ] **Step 1: Write the failing storage test**

In `tests/story_rp_engine/test_storage.py`, add `Persona` to the `story_rp_engine.core.types` import block (keep it alphabetical: `CharacterCard, GroupCard, Lorebook, LorebookEntry, Persona`), then append:

```python
@pytest.mark.anyio
async def test_persona_crud(make_store):
    store = make_store()
    persona = Persona(persona_id="sam", name="Sam", description="A wandering cartographer.")
    await store.save_persona("sam", persona)

    assert await store.get_persona("sam") == persona
    assert await store.list_personas() == {"sam": persona}

    assert await store.delete_persona("sam") is True
    assert await store.get_persona("sam") is None
    assert await store.delete_persona("sam") is False
```

- [ ] **Step 2: Write the failing API test**

Append to `tests/story_rp_engine/test_api.py`:

```python
def test_persona_crud(tmp_path):
    client = TestClient(create_app(store=EngineStore(storage_dir=str(tmp_path)), config=EngineConfig()))
    sam = {"persona_id": "sam", "name": "Sam", "description": "A wandering cartographer."}

    assert client.post("/api/v1/personas", json=sam).json() == {"status": "saved", "persona_id": "sam"}
    assert client.get("/api/v1/personas/sam").json() == sam
    assert list(client.get("/api/v1/personas").json()) == ["sam"]
    assert client.post("/api/v1/personas", json={**sam, "persona_id": "../evil"}).status_code == 400

    assert client.delete("/api/v1/personas/sam").json() == {"status": "deleted", "persona_id": "sam"}
    assert client.get("/api/v1/personas/sam").status_code == 404
    assert client.delete("/api/v1/personas/sam").status_code == 404
```

- [ ] **Step 3: Run both tests to verify they fail**

Run: `uv run pytest tests/story_rp_engine/test_storage.py::test_persona_crud tests/story_rp_engine/test_api.py::test_persona_crud -v`
Expected: collection error / FAIL with `ImportError: cannot import name 'Persona'`.

- [ ] **Step 4: Add the model**

In `src/story_rp_engine/core/types.py`, after the `GroupCard` class:

```python
class Persona(BaseModel):
    """Who the user plays as in roleplay and group chats."""

    persona_id: str
    name: str
    description: str = Field(default="", description="Added to the character prompts; {{user}} and {{char}} macros work.")
```

- [ ] **Step 5: Add the storage collection**

In `src/story_rp_engine/storage/store.py`:

1. Import: `from story_rp_engine.core.types import CharacterCard, GroupCard, Lorebook, Persona`
2. After `groups_table`, add:

```python
personas_table = Table(
    "story_rp_personas",
    _metadata,
    Column("persona_id", String(MAX_DB_KEY_LENGTH), primary_key=True),
    Column("persona_json", Text, nullable=False),
    Column("updated_at", DateTime),
)
```

3. `_SqlTables` docstring: `"""Creates the character, lorebook, group and persona tables on first use."""`
4. `EngineStore` docstring: replace the first two sentences' lists so it reads `Persists ADK sessions, character cards, lorebooks, groups and personas.` and `Without one, characters, lorebooks, groups and personas are JSON files under storage_dir and sessions go to a local SQLite file.`
5. In `__init__`, after `self.groups_dir = ...`: `self.personas_dir = os.path.join(storage_dir, "personas")`
6. In the file branch, after `self._groups = _FileCollection(...)`: `self._personas = _FileCollection(self.personas_dir, Persona)`
7. In the SQL branch, after `self._groups = _SqlCollection(...)`: `self._personas = _SqlCollection(tables, personas_table, "persona_id", "persona_json", Persona)`
8. After `delete_group`, add:

```python
    async def save_persona(self, persona_id: str, persona: Persona) -> None:
        await self._personas.put(_sanitize_key(persona_id), persona)

    async def get_persona(self, persona_id: str) -> Optional[Persona]:
        return await self._personas.get(_sanitize_key(persona_id))

    async def list_personas(self) -> Dict[str, Persona]:
        return await self._personas.list()

    async def delete_persona(self, persona_id: str) -> bool:
        """Deletes a persona; returns False if it did not exist."""
        return await self._personas.delete(_sanitize_key(persona_id))
```

- [ ] **Step 6: Add the routes**

Create `src/story_rp_engine/api/routes_persona.py`:

```python
from fastapi import APIRouter, HTTPException, Request
from story_rp_engine.core.types import Persona
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Personas"])


@router.post("/personas")
async def save_persona(persona: Persona, request: Request):
    try:
        await request.app.state.store.save_persona(persona.persona_id, persona)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "persona_id": persona.persona_id}


@router.get("/personas")
async def list_personas(request: Request):
    return await request.app.state.store.list_personas()


@router.get("/personas/{persona_id}")
async def get_persona(persona_id: str, request: Request):
    try:
        persona = await request.app.state.store.get_persona(persona_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@router.delete("/personas/{persona_id}")
async def delete_persona(persona_id: str, request: Request):
    try:
        clean_id = _sanitize_key(persona_id)
        deleted = await request.app.state.store.delete_persona(clean_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"status": "deleted", "persona_id": clean_id}
```

In `src/story_rp_engine/api/app.py`, add `from story_rp_engine.api.routes_persona import router as persona_router` after the `routes_lorebook` import, and `app.include_router(persona_router)` after `app.include_router(group_router)`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/story_rp_engine/test_storage.py tests/story_rp_engine/test_api.py -q`
Expected: all pass (the Postgres parametrization of `test_persona_crud` is skipped unless `STORY_RP_TEST_PG_URL` is set).

- [ ] **Step 8: Update docs**

1. `CLAUDE.md`, the Storage bullet: ``the `story_rp_characters`/`story_rp_lorebooks`/`story_rp_groups` tables`` → ``the `story_rp_characters`/`story_rp_lorebooks`/`story_rp_groups`/`story_rp_personas` tables``, and `characters, lorebooks and groups to JSON files` → `characters, lorebooks, groups and personas to JSON files`.
2. `.agents/skills/story-rp-backend/SKILL.md`: after the `routes_group` router bullet add `` - `story_rp_engine.api.routes_persona`: Persona CRUD (`/api/v1/personas`). ``; in the storage-directory bullet, `` `groups/`, and `sessions.db` `` → `` `groups/`, `personas/`, and `sessions.db` ``.
3. `.agents/skills/story-rp-backend/references/database-session-storage.md`: in tier 1, replace ``and groups (`groups/{group_id}.json`)`` with ``groups (`groups/{group_id}.json`) and personas (`personas/{persona_id}.json`)``, and the table list with `` `story_rp_characters`, `story_rp_lorebooks`, `story_rp_groups` and `story_rp_personas` tables ``; in the mermaid graph add `    Store --> DiskPersonas["Disk: personas/*.json"]` after the `DiskGroups` line; in the directory tree add after the `groups/` block:

```
├── personas/
│   └── sam.json
```

4. `.agents/skills/story-rp-backend/references/api-endpoints-and-sse.md`: in the router mermaid graph add `    FastAPI --> Personas["Persona Router (/api/v1/personas)"]` after the `Lore` line; insert a new section before `### 1.5 System Endpoints` and renumber that one to `1.6`:

```markdown
### 1.5 Persona Endpoints (`/api/v1/personas`)

A persona is who the user plays as in roleplay and group chats: a `name` and a `description` that goes into the character prompts.

- `POST /api/v1/personas`: Saves a `Persona` (`persona_id`, `name`, `description`). Returns `{"status": "saved", "persona_id": ...}`; `400` for an invalid `persona_id`.
- `GET /api/v1/personas`: Returns all personas keyed by `persona_id`.
- `GET /api/v1/personas/{persona_id}`: Retrieves one persona (`404` if missing).
- `DELETE /api/v1/personas/{persona_id}`: Deletes the persona (`404` if missing). Chats that used it keep their copy in session state.
```

- [ ] **Step 9: Commit**

```bash
git add src/story_rp_engine/core/types.py src/story_rp_engine/storage/store.py src/story_rp_engine/api/routes_persona.py src/story_rp_engine/api/app.py tests/story_rp_engine/test_storage.py tests/story_rp_engine/test_api.py CLAUDE.md .agents/skills/story-rp-backend
git commit -m "add saved user personas with crud api

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `<user_persona>` section in the RP and group prompts

**Files:**
- Modify: `src/story_rp_engine/rp/prompt_builder.py:15-41`
- Modify: `src/story_rp_engine/group/prompt_builder.py:21-47`
- Test: `tests/story_rp_engine/test_prompt_builder.py`, `tests/story_rp_engine/test_group_prompt_builder.py`

**Interfaces:**
- Produces: `build_rp_system_instruction(card, active_lore=None, user_name="User", greeting=None, user_persona: str = "") -> str` and `build_group_system_instruction(card, group, cards, user_name="User", user_persona: str = "") -> str`. Both add `<user_persona>\n{filled}\n</user_persona>` right after the `<scenario>` section when `user_persona` is non-empty.

- [ ] **Step 1: Write the failing RP prompt tests**

Append to `tests/story_rp_engine/test_prompt_builder.py` (uses the existing `_aria()` helper):

```python
def test_build_rp_system_instruction_adds_user_persona_after_scenario():
    card = _aria().model_copy(update={"scenario": "A border fort."})
    instruction = build_rp_system_instruction(
        card, user_name="Sam", user_persona="{{user}} is a cartographer who owes {{char}} money."
    )
    assert "<user_persona>\nSam is a cartographer who owes Aria money.\n</user_persona>" in instruction
    assert instruction.index("<scenario>") < instruction.index("<user_persona>")


def test_build_rp_system_instruction_without_user_persona():
    assert "<user_persona>" not in build_rp_system_instruction(_aria(), user_name="Sam")
```

- [ ] **Step 2: Write the failing group prompt test**

Append to `tests/story_rp_engine/test_group_prompt_builder.py`:

```python
def test_group_instruction_adds_user_persona_for_each_speaker():
    persona = "{{user}} is a cartographer who owes {{char}} money."
    alice = build_group_system_instruction(ALICE, GROUP, [ALICE, BOB], user_name="Sam", user_persona=persona)
    bob = build_group_system_instruction(BOB, GROUP, [ALICE, BOB], user_name="Sam", user_persona=persona)

    assert "<user_persona>\nSam is a cartographer who owes Alice money.\n</user_persona>" in alice
    assert alice.index("<scenario>") < alice.index("<user_persona>") < alice.index("<other_characters>")
    assert "Sam is a cartographer who owes Bob money." in bob
    assert "<user_persona>" not in build_group_system_instruction(ALICE, GROUP, [ALICE, BOB], user_name="Sam")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/story_rp_engine/test_prompt_builder.py tests/story_rp_engine/test_group_prompt_builder.py -q`
Expected: 3 FAIL with `TypeError: ... got an unexpected keyword argument 'user_persona'`.

- [ ] **Step 4: Implement the RP section**

In `src/story_rp_engine/rp/prompt_builder.py`, add the parameter and the section:

```python
def build_rp_system_instruction(
    card: CharacterCard,
    active_lore: Optional[List[LorebookEntry]] = None,
    user_name: str = "User",
    greeting: Optional[str] = None,
    user_persona: str = "",
) -> str:
    """Builds the roleplay system prompt from a character card, optional lore, the user's persona and the opening greeting."""
```

and directly after the `if card.scenario:` block:

```python
    if user_persona:
        sections.append(f"<user_persona>\n{fill(user_persona)}\n</user_persona>")
```

- [ ] **Step 5: Implement the group section**

In `src/story_rp_engine/group/prompt_builder.py`, add `user_persona: str = "",` after `user_name: str = "User",` in `build_group_system_instruction`'s signature, and directly after the `if group.scenario:` block:

```python
    if user_persona:
        sections.append(f"<user_persona>\n{fill(user_persona)}\n</user_persona>")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/story_rp_engine/test_prompt_builder.py tests/story_rp_engine/test_group_prompt_builder.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/story_rp_engine/rp/prompt_builder.py src/story_rp_engine/group/prompt_builder.py tests/story_rp_engine/test_prompt_builder.py tests/story_rp_engine/test_group_prompt_builder.py
git commit -m "add user persona section to rp and group prompts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `persona_id` on chat turns

**Files:**
- Modify: `src/story_rp_engine/core/types.py` (`RPChatRequest`, `GroupChatRequest`)
- Modify: `src/story_rp_engine/api/chat_sessions.py:37-55,76-91`
- Modify: `src/story_rp_engine/rp/agent.py:24-28`
- Modify: `src/story_rp_engine/group/char_agent.py:19-23`
- Test: `tests/story_rp_engine/test_api.py`, `tests/story_rp_engine/test_group_api.py`
- Docs: `CLAUDE.md`, `.agents/skills/story-rp-backend/references/api-endpoints-and-sse.md`, `.agents/skills/character-rp/references/character-card-v2-spec.md`

**Interfaces:**
- Consumes: `Persona`, `EngineStore.get_persona` (Task 1); `user_persona` parameters of both prompt builders (Task 2).
- Produces: `persona_id: Optional[str] = None` on `RPChatRequest` and `GroupChatRequest`; session state keys `persona_id`, `user_name`, `user_persona`; `persona_id` in each item of `GET /api/v1/rp/sessions` and `GET /api/v1/group/sessions`.

- [ ] **Step 1: Write the failing RP API tests**

Append to `tests/story_rp_engine/test_api.py` (uses the existing `_recording_rp_app`, whose character `ava` has description `"Ava owes {{user}} a favor."` and whose `prompts` list holds each model call's system instruction plus message texts):

```python
SAM = {"persona_id": "sam", "name": "Sam", "description": "{{user}} is a cartographer who owes {{char}} money."}


@pytest.mark.anyio
async def test_rp_chat_persona_sets_name_and_description(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    assert client.post("/api/v1/personas", json=SAM).status_code == 200
    chat = {"char_id": "ava", "session_id": "s1", "message": "Hi"}

    assert client.post("/api/v1/rp/chat", json={**chat, "persona_id": "sam"}).status_code == 200
    assert "Ava owes Sam a favor." in prompts[-1]
    assert "Sam is a cartographer who owes Ava money." in prompts[-1]
    assert client.get("/api/v1/rp/sessions?char_id=ava").json()[0]["persona_id"] == "sam"

    # Without persona_id the chat falls back to user_name and drops the description.
    assert client.post("/api/v1/rp/chat", json={**chat, "user_name": "Alice"}).status_code == 200
    assert "Ava owes Alice a favor." in prompts[-1]
    assert "cartographer" not in prompts[-1]
    assert client.get("/api/v1/rp/sessions?char_id=ava").json()[0]["persona_id"] is None


@pytest.mark.anyio
async def test_rp_chat_uses_edited_persona_on_next_turn(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    chat = {"char_id": "ava", "session_id": "s1", "message": "Hi", "persona_id": "sam"}
    client.post("/api/v1/personas", json=SAM)
    assert client.post("/api/v1/rp/chat", json=chat).status_code == 200

    # Braces that aren't macros reach the prompt as-is; ADK's {state} templating never sees the description.
    client.post("/api/v1/personas", json={**SAM, "name": "Samantha", "description": "Carries a {lucky} coin."})
    assert client.post("/api/v1/rp/chat", json=chat).status_code == 200
    assert "Carries a {lucky} coin." in prompts[-1]
    assert "Ava owes Samantha a favor." in prompts[-1]
    assert "cartographer" not in prompts[-1]


@pytest.mark.anyio
async def test_rp_chat_unknown_persona_returns_404(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    res = client.post("/api/v1/rp/chat", json={"char_id": "ava", "session_id": "s1", "message": "Hi", "persona_id": "ghost"})
    assert res.status_code == 404
    assert res.json()["detail"] == "Persona not found: ghost"
    assert prompts == []
```

- [ ] **Step 2: Write the failing group API test**

Append to `tests/story_rp_engine/test_group_api.py` (uses the existing `_group_client`, `_chat` and the `group_models` fixture from `conftest.py`):

```python
@pytest.mark.anyio
async def test_group_chat_persona_reaches_member_prompts(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["alice"]}
    sam = {"persona_id": "sam", "name": "Sam", "description": "{{user}} owes {{char}} money."}
    assert client.post("/api/v1/personas", json=sam).status_code == 200

    assert _chat(client, "Hello!", persona_id="sam").status_code == 200
    instruction = group_models.requests["Alice"][-1].config.system_instruction
    assert instruction.startswith("You are Alice in a group roleplay with Sam, Bob.")
    assert "Sam owes Alice money." in instruction
    assert "group roleplay between Sam and" in group_models.requests["speaker_selector"][-1].config.system_instruction
    assert client.get("/api/v1/group/sessions?group_id=tavern").json()[0]["persona_id"] == "sam"

    res = _chat(client, "Hello?", persona_id="ghost")
    assert res.status_code == 404
    assert res.json()["detail"] == "Persona not found: ghost"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/story_rp_engine/test_api.py -k persona tests/story_rp_engine/test_group_api.py -q`
Expected: the four new chat tests FAIL (the persona is ignored: `"Ava owes Sam a favor."` not in the prompt; the unknown persona returns `200`). `test_persona_crud` from Task 1 still passes.

- [ ] **Step 4: Add `persona_id` to the chat requests**

In `src/story_rp_engine/core/types.py`, add to both `RPChatRequest` and `GroupChatRequest`, directly after their `user_name` line:

```python
    persona_id: Optional[str] = Field(default=None, description="Plays as this saved persona; overrides user_name.")
```

- [ ] **Step 5: Resolve the persona in `chat_state_delta`**

In `src/story_rp_engine/api/chat_sessions.py`, replace `chat_state_delta` with:

```python
async def chat_state_delta(req, request: Request, **fields) -> dict:
    """Session state changes for a chat turn; lorebooks are copied in only when lorebook_ids is sent.

    A persona_id sets the user's name and description from that saved persona, re-read every turn so edits apply;
    without one the name comes from user_name and the description is cleared. `fields` adds the chat's own keys
    (char_id and greeting, or group_id). The owner ID and last_message are read by the history list, which gets
    session state but no events.
    """
    persona = None
    if req.persona_id:
        persona = await request.app.state.store.get_persona(req.persona_id)
        if persona is None:
            raise HTTPException(status_code=404, detail=f"Persona not found: {req.persona_id}")
    state_delta = {
        "authors_note": req.authors_note,
        "persona_id": req.persona_id,
        "user_name": persona.name if persona else (req.user_name or "User"),
        "user_persona": persona.description if persona else "",
        **fields,
        "last_message": req.message[:80],
    }
    if req.lorebook_ids is not None:
        state_delta["lorebook"] = await load_lorebooks(request, req.lorebook_ids)
        # Read by the history list, so reopening a chat reselects its lorebooks.
        state_delta["lorebook_ids"] = req.lorebook_ids
    return state_delta
```

In `list_chat_sessions`, add `"persona_id": s.state.get("persona_id"),` after the `"user_name"` line.

- [ ] **Step 6: Pass `user_persona` to the prompt builders**

In `src/story_rp_engine/rp/agent.py`, the instruction lambda becomes:

```python
        instruction=lambda ctx: build_rp_system_instruction(
            card,
            user_name=ctx.state.get("user_name") or user_name,
            greeting=ctx.state.get("greeting"),
            user_persona=ctx.state.get("user_persona") or "",
        ),
```

In `src/story_rp_engine/group/char_agent.py`, the instruction lambda becomes:

```python
        instruction=lambda ctx: build_group_system_instruction(
            card,
            group,
            cards,
            user_name=ctx.state.get("user_name") or "User",
            user_persona=ctx.state.get("user_persona") or "",
        ),
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/story_rp_engine -q`
Expected: all pass.

- [ ] **Step 8: Update docs**

1. `CLAUDE.md`, Roleplay bullet: replace the sentence ``Session state keys (`user_name`, `greeting`, `authors_note`, `lorebook`) are set through `state_delta` in `api/chat_sessions.chat_state_delta`.`` with ``Session state keys (`user_name`, `user_persona`, `persona_id`, `greeting`, `authors_note`, `lorebook`) are set through `state_delta` in `api/chat_sessions.chat_state_delta`, which resolves the request's `persona_id` into the saved persona's name and description every turn; the description goes in the system prompt as `<user_persona>`.``
2. `.agents/skills/story-rp-backend/references/api-endpoints-and-sse.md`:
   - Under `POST /api/v1/rp/chat`'s request body, after the `user_name` bullet add: `` - `persona_id` (str, optional): Saved persona to play as; sets the user's name and adds its description to the prompt, overriding `user_name`. `404` if it doesn't exist. ``
   - In the `POST /api/v1/group/chat/stream` bullet: `` `authors_note`, `user_name`, `lorebook_ids` `` → `` `authors_note`, `user_name`, `persona_id`, `lorebook_ids` `` and `` `404` if the group or lorebook is missing `` → `` `404` if the group, lorebook or persona is missing ``.
3. `.agents/skills/character-rp/references/character-card-v2-spec.md`, the `{{user}}` row of the macro table: replace ``User's configured name (`user_name`, defaults to `"User"`)`` with ``The selected persona's name, else `user_name` (defaults to `"User"`)``.

- [ ] **Step 9: Commit**

```bash
git add src/story_rp_engine/core/types.py src/story_rp_engine/api/chat_sessions.py src/story_rp_engine/rp/agent.py src/story_rp_engine/group/char_agent.py tests/story_rp_engine/test_api.py tests/story_rp_engine/test_group_api.py CLAUDE.md .agents/skills
git commit -m "play rp and group chats as a saved persona

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Personas tab

**Files:**
- Modify: `src/story_rp_engine/web/index.html` (nav after the Lorebooks button at lines 70-76; new section after the Lorebooks section, before `</main>`)
- Modify: `src/story_rp_engine/web/app.js` (data, methods after `deleteGroup`, `mounted`)
- Test: `tests/test_frontend_integration.py`
- Docs: `.agents/skills/story-rp-frontend/SKILL.md`, `.agents/skills/story-rp-frontend/references/ui-components-and-modes.md`

**Interfaces:**
- Consumes: `/api/v1/personas` routes (Task 1).
- Produces: Vue data `personas` (`{persona_id: Persona}`), `selectedPersonaId`, `personaForm`; methods `loadPersonas()`, `selectPersona(id)`, `newPersona()`, `savePersona()`, `deletePersona(id)`. Task 5 reads `personas` and adds to `deletePersona`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_frontend_integration.py`:

```python
def test_personas_tab_is_served(client):
    html = client.get("/").text
    assert "activeTab === 'personas'" in html
    assert "savePersona" in html
    js = client.get("/app.js").text
    assert "'/api/v1/personas'" in js
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_frontend_integration.py::test_personas_tab_is_served -v`
Expected: FAIL on `assert "activeTab === 'personas'" in html`.

- [ ] **Step 3: Add the nav button**

In `src/story_rp_engine/web/index.html`, after the Lorebooks `<button>` (the one whose `<span>` is `Lorebooks`) and before `</nav>`:

```html
        <button
          @click="activeTab = 'personas'"
          :class="activeTab === 'personas' ? 'bg-indigo-600 text-white font-medium shadow-sm' : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'"
          class="flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs transition">
          <i data-lucide="user" class="w-3.5 h-3.5"></i>
          <span>Personas</span>
        </button>
```

- [ ] **Step 4: Add the tab section**

In `index.html`, after the Lorebooks `</section>` and before `</main>`:

```html

      <!-- ================================================================= -->
      <!-- PERSONAS MANAGEMENT WORKBENCH                                     -->
      <!-- ================================================================= -->
      <section v-show="activeTab === 'personas'" class="w-full h-full flex flex-col md:flex-row overflow-hidden">

        <!-- Left Column: Persona Directory -->
        <aside class="w-full md:w-80 lg:w-96 flex-shrink-0 border-r border-neutral-800 bg-neutral-900/60 overflow-y-auto p-4 space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-neutral-800">
            <h2 class="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center space-x-1.5">
              <i data-lucide="user" class="w-3.5 h-3.5 text-indigo-400"></i>
              <span>Personas Directory</span>
            </h2>
            <span class="text-[10px] font-mono text-neutral-500">{{ Object.keys(personas).length }} saved</span>
          </div>

          <button
            @click="newPersona"
            class="w-full flex items-center justify-center space-x-1 px-2.5 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition shadow-sm">
            <i data-lucide="plus" class="w-3 h-3"></i>
            <span>New Persona</span>
          </button>

          <div class="space-y-2">
            <div v-if="Object.keys(personas).length === 0" class="text-center py-6 text-xs text-neutral-500">
              No personas yet.
            </div>
            <div
              v-for="(p, id) in personas"
              :key="id"
              @click="selectPersona(id)"
              :class="selectedPersonaId === id ? 'border-indigo-500 bg-indigo-950/20' : 'border-neutral-800 bg-neutral-950/50 hover:border-neutral-700'"
              class="p-3 rounded-lg border cursor-pointer transition space-y-1.5 select-text">
              <span class="font-medium text-xs text-neutral-200">{{ p.name || id }}</span>
              <p class="text-[11px] text-neutral-400 line-clamp-2 leading-relaxed">{{ p.description }}</p>
            </div>
          </div>
        </aside>

        <!-- Right Column: Persona Editor -->
        <div class="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 bg-neutral-950">
          <div class="flex items-center justify-between pb-4 border-b border-neutral-800">
            <div>
              <h2 class="text-sm md:text-base font-bold text-neutral-100 flex items-center space-x-2">
                <span>{{ personaForm.name || 'New Persona' }}</span>
                <span v-if="selectedPersonaId" class="text-xs font-mono text-neutral-500">({{ selectedPersonaId }})</span>
              </h2>
              <p class="text-xs text-neutral-400">Who you play as in Roleplay and Group Chat.</p>
            </div>
            <div class="flex items-center space-x-2">
              <button
                v-if="selectedPersonaId"
                @click="deletePersona(selectedPersonaId)"
                class="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-rose-600/80 hover:bg-rose-600 text-white text-xs font-medium transition shadow-sm">
                <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                <span>Delete</span>
              </button>
              <button
                @click="savePersona"
                class="flex items-center space-x-1.5 px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition shadow-sm">
                <i data-lucide="save" class="w-3.5 h-3.5"></i>
                <span>Save Persona</span>
              </button>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="space-y-1.5">
              <label class="block text-xs font-medium text-neutral-300">Persona ID (Slug) <span class="text-rose-400">*</span></label>
              <input
                type="text"
                v-model="personaForm.persona_id"
                placeholder="e.g. sam_cartographer"
                class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 font-mono focus:border-indigo-500">
            </div>
            <div class="space-y-1.5">
              <label class="block text-xs font-medium text-neutral-300">Name <span class="text-rose-400">*</span></label>
              <input
                type="text"
                v-model="personaForm.name"
                placeholder="e.g. Sam"
                class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
            </div>
          </div>

          <div class="space-y-1.5 pb-8">
            <label class="block text-xs font-medium text-neutral-300">Description</label>
            <textarea
              v-model="personaForm.description"
              rows="6"
              placeholder="Who you are in the scene; the characters see this. e.g. {{user}} is a wandering cartographer who owes {{char}} a favor."
              class="w-full bg-neutral-900 border border-neutral-800 rounded-md px-3 py-2 text-xs text-neutral-200 placeholder-neutral-600 focus:border-indigo-500"></textarea>
          </div>
        </div>

      </section>
```

(`{{user}}` inside the `placeholder` attribute is safe: Vue only interpolates text nodes, and the Characters tab already does this.)

- [ ] **Step 5: Add state and methods**

In `src/story_rp_engine/web/app.js`:

1. The `activeTab` comment becomes `// 'roleplay' | 'group' | 'story' | 'characters' | 'groups' | 'lorebooks' | 'personas'`.
2. After the Groups Management State block in `data()`:

```js
      // =======================================================================
      // Personas Management State
      // =======================================================================
      personas: {}, // Object mapping persona_id -> Persona
      selectedPersonaId: null,
      personaForm: { persona_id: '', name: '', description: '' },
```

3. After the `deleteGroup` method:

```js
    async loadPersonas() {
      try {
        const res = await fetch('/api/v1/personas');
        if (res.ok) this.personas = await res.json();
      } catch (err) {
        console.error('Failed to load personas:', err);
      }
      this.refreshIcons();
    },

    selectPersona(id) {
      const p = this.personas[id];
      if (!p) return;
      this.selectedPersonaId = id;
      this.personaForm = { persona_id: p.persona_id, name: p.name || '', description: p.description || '' };
      this.refreshIcons();
    },

    newPersona() {
      this.selectedPersonaId = null;
      this.personaForm = { persona_id: '', name: '', description: '' };
      this.refreshIcons();
    },

    async savePersona() {
      const personaId = (this.personaForm.persona_id || '').trim();
      const name = (this.personaForm.name || '').trim();
      if (!personaId || !name) {
        this.showToast('Persona ID and name are required.', 'error');
        return;
      }
      try {
        const res = await fetch('/api/v1/personas', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ persona_id: personaId, name: name, description: this.personaForm.description || '' }),
        });
        if (res.ok) {
          this.showToast(`Persona "${name}" saved!`, 'success');
          this.selectedPersonaId = personaId;
          await this.loadPersonas();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to save persona' }));
          this.showToast(err.detail || 'Save failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while saving persona', 'error');
      }
    },

    async deletePersona(id) {
      if (!id || !confirm(`Are you sure you want to delete persona "${id}"?`)) return;
      try {
        const res = await fetch(`/api/v1/personas/${encodeURIComponent(id)}`, { method: 'DELETE' });
        if (res.ok) {
          this.showToast(`Persona "${id}" deleted.`, 'success');
          if (this.selectedPersonaId === id) this.newPersona();
          await this.loadPersonas();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to delete' }));
          this.showToast(err.detail || 'Delete failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while deleting persona', 'error');
      }
    },
```

4. In `mounted()`, add `this.loadPersonas();` after `this.loadGroups();`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_frontend_integration.py tests/test_web_mount.py -q`
Expected: all pass.

- [ ] **Step 7: Update docs**

1. `.agents/skills/story-rp-frontend/SKILL.md`:
   - frontmatter `description`: `characters, groups, lorebooks)` → `characters, groups, lorebooks, personas)`.
   - `activeTab` list: append `` `'personas'` `` after `` `'lorebooks'` ``.
   - After the **Groups** bullet group in section 3, add:

```markdown
- **Personas** (Personas tab):
  - Load via `GET /api/v1/personas` into `personas` (keyed by `persona_id`); save via `POST /api/v1/personas` and delete via `DELETE /api/v1/personas/{id}`.
  - `personaForm` holds `persona_id`, `name` and `description` (the description goes into the character prompts; `{{user}}`/`{{char}}` macros work).
```

   - Reference Guides line: `the six tabs (Roleplay, Group Chat, Story Co-Pilot, Characters, Groups, Lorebooks)` → `the seven tabs (Roleplay, Group Chat, Story Co-Pilot, Characters, Groups, Lorebooks, Personas)`, and `lorebook and group editors` → `lorebook, group and persona editors`.
2. `.agents/skills/story-rp-frontend/references/ui-components-and-modes.md`:
   - In the top mermaid graph add `    Main --> TabPersonas["Personas Directory & Editor (activeTab === 'personas')"]` after the `Tab4` line.
   - Navigation Tabs: `Six tab buttons` → `Seven tab buttons`, and add the line ``     - `personas` (`lucide="user"`, Personas editor)`` after the `lorebooks` line.
   - After section 8 (Groups Management), insert before the `---` that precedes section 9, and renumber section 9 to 10:

```markdown
---

## 9. Personas Management (`personas`)

Editor for who the user plays as in Roleplay and Group Chat.

### Layout & Features
- **Sidebar Directory**: one card per persona (name and a two-line description preview); **New Persona (`newPersona()`)** clears the form. No search box.
- **Form Editor** (`personaForm`): `persona_id` (slug), `name`, and `description` (added to the character prompts; `{{user}}` and `{{char}}` macros work).
- **Actions**: Save Persona (`POST /api/v1/personas`), Delete (`DELETE /api/v1/personas/{id}`).
```

- [ ] **Step 8: Run the skills check**

Run: `uv run pytest tests/test_agent_skills.py -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/story_rp_engine/web/index.html src/story_rp_engine/web/app.js tests/test_frontend_integration.py .agents/skills/story-rp-frontend
git commit -m "add personas tab

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Persona dropdowns in the RP and group chat sidebars

**Files:**
- Modify: `src/story_rp_engine/web/index.html:224-232` (RP "User Persona Name"), `:306` (RP user label), `:489-497` (group "User Persona Name")
- Modify: `src/story_rp_engine/web/app.js` (data `rpUserName`/`groupUserName`, `openRPSession`, `sendRPMessage` payload, `groupSpeakerName`, `openGroupSession`, `sendGroupMessage` payload, `deletePersona`)
- Test: `tests/test_frontend_integration.py`
- Docs: `.agents/skills/story-rp-frontend/references/ui-components-and-modes.md`, `.agents/skills/story-rp-frontend/references/sse-event-client.md`

**Interfaces:**
- Consumes: `personas`, `deletePersona` (Task 4); `persona_id` on the chat requests and in session list items (Task 3).
- Produces: Vue data `rpPersonaId`, `groupPersonaId` (`''` = no persona); method `personaName(personaId) -> string`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_frontend_integration.py`:

```python
def test_chat_tabs_pick_a_saved_persona(client):
    html = client.get("/").text
    assert 'v-model="rpPersonaId"' in html
    assert 'v-model="groupPersonaId"' in html
    js = client.get("/app.js").text
    assert "persona_id: this.rpPersonaId" in js
    assert "persona_id: this.groupPersonaId" in js
    # The free-text name fields are gone; the name comes from the persona.
    assert "rpUserName" not in js + html
    assert "groupUserName" not in js + html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_frontend_integration.py::test_chat_tabs_pick_a_saved_persona -v`
Expected: FAIL on `assert 'v-model="rpPersonaId"' in html`.

- [ ] **Step 3: Replace the sidebar inputs**

In `src/story_rp_engine/web/index.html`, replace the RP block

```html
          <!-- User Name / Persona -->
          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">User Persona Name</label>
            <input
              type="text"
              v-model="rpUserName"
              placeholder="User"
              class="w-full bg-neutral-950 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
          </div>
```

with

```html
          <!-- User Persona -->
          <div class="space-y-1.5">
            <label class="block text-xs font-medium text-neutral-300">Persona</label>
            <select
              v-model="rpPersonaId"
              class="w-full bg-neutral-950 border border-neutral-800 rounded-md px-3 py-1.5 text-xs text-neutral-200 focus:border-indigo-500">
              <option value="">User (no persona)</option>
              <option v-for="(p, id) in personas" :key="id" :value="id">{{ p.name }}</option>
            </select>
          </div>
```

and the group block (same markup with `v-model="groupUserName"`) with the same `<select>` using `v-model="groupPersonaId"`.

In the RP message header, `{{ msg.role === 'user' ? (rpUserName || 'User') : (selectedCharName || 'Assistant') }}` → `{{ msg.role === 'user' ? personaName(rpPersonaId) : (selectedCharName || 'Assistant') }}`.

- [ ] **Step 4: Update app.js**

In `src/story_rp_engine/web/app.js`:

1. Data: `rpUserName: 'User',` → `rpPersonaId: '', // '' plays as "User"`; `groupUserName: 'User',` → `groupPersonaId: '', // '' plays as "User"`.
2. Add a method next to `characterName`:

```js
    personaName(personaId) {
      const p = this.personas[personaId];
      return p ? p.name : 'User';
    },
```

3. `openRPSession`: `this.rpUserName = s.user_name || 'User';` →

```js
        // A deleted persona falls back to "no persona".
        this.rpPersonaId = this.personas[s.persona_id] ? s.persona_id : '';
```

4. `openGroupSession`: `this.groupUserName = s.user_name || 'User';` → `this.groupPersonaId = this.personas[s.persona_id] ? s.persona_id : '';`
5. `sendRPMessage` payload: `user_name: this.rpUserName ? this.rpUserName.trim() : 'User',` →

```js
        // undefined drops the key, so the backend plays as "User".
        persona_id: this.rpPersonaId || undefined,
```

6. `sendGroupMessage` payload: `user_name: this.groupUserName ? this.groupUserName.trim() : 'User',` → `persona_id: this.groupPersonaId || undefined,`
7. `groupSpeakerName`: `if (msg.role === 'user') return this.groupUserName || 'User';` → `if (msg.role === 'user') return this.personaName(this.groupPersonaId);`
8. `deletePersona`, inside `if (res.ok) {` after `if (this.selectedPersonaId === id) this.newPersona();`:

```js
          if (this.rpPersonaId === id) this.rpPersonaId = '';
          if (this.groupPersonaId === id) this.groupPersonaId = '';
```

Then confirm nothing else references the old names: `grep -n "rpUserName\|groupUserName" src/story_rp_engine/web/*` prints nothing.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_frontend_integration.py tests/test_web_mount.py -q`
Expected: all pass.

- [ ] **Step 6: Check it in the browser**

Run: `STORY_RP_SKIP_WARMUP=1 ./run_sh/run_story_rp_backend.sh` and open `http://localhost:8000/`. Without a model server only the UI flow can be checked; with one, also check the replies.
- Personas tab: create `sam` / `Sam` / `{{user}} is a cartographer.`; it appears in the list; edit and save; reload the page and it's still there.
- Roleplay tab: the Persona dropdown lists "User (no persona)" and "Sam"; pick Sam, send a message; the user bubble is labelled "Sam". Start a new chat, switch to "User (no persona)", open the Sam chat from history: the dropdown shows Sam again.
- Group Chat tab: same dropdown; user bubbles are labelled with the persona name.
- Delete Sam in the Personas tab: both chat dropdowns go back to "User (no persona)".
- Browser devtools console shows no errors.

- [ ] **Step 7: Update docs**

1. `.agents/skills/story-rp-frontend/references/ui-components-and-modes.md`:
   - RP sidebar mermaid: `- User Persona Name (rpUserName)` → `- Persona (rpPersonaId)`.
   - RP sidebar controls: replace the ``**User Persona Name (`rpUserName`)**`` bullet with ``- **Persona (`rpPersonaId`)**: `<select>` over `personas` plus "User (no persona)" (`''`). Sent as `persona_id` (left out when `''`); user bubbles are labelled `personaName(rpPersonaId)`. Reopening a chat reselects its `persona_id` if that persona still exists.``
   - Group sidebar: replace ``**User Persona Name** (`groupUserName`)`` with ``**Persona** (`groupPersonaId`, same as RP)``.
   - Group chat feed: replace ``Bubble labels come from `groupSpeakerName(msg)`: the user name,`` with ``Bubble labels come from `groupSpeakerName(msg)`: `personaName(groupPersonaId)` for the user,``.
2. `.agents/skills/story-rp-frontend/references/sse-event-client.md`, the Request Model row: in both the RP and Group cells, `` `user_name` `` → `` `persona_id` ``.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (Postgres tests skipped without `STORY_RP_TEST_PG_URL`).

- [ ] **Step 9: Commit**

```bash
git add src/story_rp_engine/web/index.html src/story_rp_engine/web/app.js tests/test_frontend_integration.py .agents/skills/story-rp-frontend
git commit -m "pick a saved persona in rp and group chat

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
