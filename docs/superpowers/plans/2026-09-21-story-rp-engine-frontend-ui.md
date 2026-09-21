# Story RP Engine Frontend UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and integrate a zero-build modern frontend workbench for `story_rp_engine` with complete editability of Character Cards, Lorebooks, Roleplay streaming chat with turn deletion, and Story Co-Pilot prose expansion.

**Architecture:** A zero-build Single Page Application (SPA) served directly by FastAPI via a static mount at `/`. Built with Vue 3 and Tailwind CSS loaded via CDN, connecting to the backend via standard REST endpoints for CRUD and Server-Sent Events (SSE) for streaming text generation.

**Tech Stack:** FastAPI, Python 3.11, Vue 3 (CDN global/ESM), Tailwind CSS (CDN), Lucide Icons (CDN), Marked.js (CDN), pytest, httpx.

**Spec:** [`docs/superpowers/specs/2026-09-21-story-rp-engine-frontend-ui-design.md`](file:///home/tangc/slm-vlm-post-training/docs/superpowers/specs/2026-09-21-story-rp-engine-frontend-ui-design.md)

## Global Constraints
- **Zero build dependencies**: No Node.js or `npm` required; all frontend dependencies are delivered via modern CDNs or pure browser-native ESM.
- **Data Model Fidelity**: Every field in [`CharacterCard`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L5-L18), [`Lorebook`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L27-L31), [`StoryRequest`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L39-L48), and [`RPChatRequest`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L50-L57) must have a corresponding, editable input in the UI.
- **Turn Deletion Consistency**: Deleting a turn from the UI must sync with the backend session event history so memory stays accurate.
- **Test Quality**: All backend endpoints and file serving routes must be covered with automated unit/integration tests running under `uv run pytest`.

---

### Task 1: Backend Lorebook CRUD & Character Deletion Endpoints

**Files:**
- Create: `src/story_rp_engine/api/routes_lorebook.py`
- Modify: `src/story_rp_engine/api/routes_rp.py:30-45`
- Modify: `src/story_rp_engine/api/app.py:100-115`
- Test: `tests/test_api_lorebook_character_crud.py`

**Interfaces:**
- Consumes: [`EngineStore`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/storage/store.py) (`list_lorebooks`, `get_lorebook`, `save_lorebook`, `list_characters`, `get_character`, `save_character`)
- Produces:
  - `GET /api/v1/lorebooks` -> `Dict[str, Lorebook]`
  - `GET /api/v1/lorebooks/{lorebook_id}` -> `Lorebook`
  - `POST /api/v1/lorebooks` -> `{"status": "saved", "lorebook_id": str}`
  - `DELETE /api/v1/lorebooks/{lorebook_id}` -> `{"status": "deleted"}`
  - `DELETE /api/v1/characters/{char_id}` -> `{"status": "deleted"}`

- [ ] **Step 1: Write the failing test for Lorebook CRUD and Character deletion**

Create `tests/test_api_lorebook_character_crud.py`:
```python
import pytest
from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.config import EngineConfig


@pytest.fixture
def client(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path / "engine_data"))
    store = EngineStore(storage_dir=config.storage_dir)
    app = create_app(store=store, config=config)
    return TestClient(app)


def test_character_deletion(client):
    card_data = {
        "char_id": "test_char",
        "name": "Test Character",
        "description": "A character for testing deletion",
    }
    create_resp = client.post("/api/v1/characters", json=card_data)
    assert create_resp.status_code == 200

    del_resp = client.delete("/api/v1/characters/test_char")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "deleted", "char_id": "test_char"}

    get_resp = client.get("/api/v1/characters/test_char")
    assert get_resp.status_code == 404


def test_lorebook_crud(client):
    lb_data = {
        "name": "Eldoria World",
        "description": "World lore for Eldoria",
        "entries": [
            {
                "keys": ["eldoria", "kingdom"],
                "content": "A high fantasy kingdom.",
                "insertion_order": 100,
                "enabled": True,
            }
        ],
    }
    save_resp = client.post("/api/v1/lorebooks", json=lb_data)
    assert save_resp.status_code == 200
    lb_id = save_resp.json()["lorebook_id"]

    get_resp = client.get(f"/api/v1/lorebooks/{lb_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Eldoria World"

    list_resp = client.get("/api/v1/lorebooks")
    assert list_resp.status_code == 200
    assert lb_id in list_resp.json()

    del_resp = client.delete(f"/api/v1/lorebooks/{lb_id}")
    assert del_resp.status_code == 200

    get_after_del = client.get(f"/api/v1/lorebooks/{lb_id}")
    assert get_after_del.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_lorebook_character_crud.py -v`  
Expected: FAIL (endpoints `/api/v1/lorebooks` and `DELETE /api/v1/characters/{char_id}` not found 404/405).

- [ ] **Step 3: Implement Lorebook routes and Character deletion**

1. Create `src/story_rp_engine/api/routes_lorebook.py`:
```python
import os
import re
from fastapi import APIRouter, HTTPException, Request
from story_rp_engine.core.types import Lorebook
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Lorebooks"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
    return slug.strip("_") or "lorebook"


@router.post("/lorebooks")
def save_lorebook(lorebook: Lorebook, request: Request):
    store = request.app.state.store
    lb_id = _slugify(lorebook.name)
    try:
        store.save_lorebook(lb_id, lorebook)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "lorebook_id": lb_id}


@router.get("/lorebooks")
def list_lorebooks(request: Request):
    store = request.app.state.store
    return store.list_lorebooks()


@router.get("/lorebooks/{lorebook_id}")
def get_lorebook(lorebook_id: str, request: Request):
    store = request.app.state.store
    try:
        lb = store.get_lorebook(lorebook_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not lb:
        raise HTTPException(status_code=404, detail="Lorebook not found")
    return lb


@router.delete("/lorebooks/{lorebook_id}")
def delete_lorebook(lorebook_id: str, request: Request):
    store = request.app.state.store
    try:
        clean_id = _sanitize_key(lorebook_id)
        path = os.path.join(store.lorebooks_dir, f"{clean_id}.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Lorebook not found")
        os.remove(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted", "lorebook_id": clean_id}
```

2. Add `delete_character` to `src/story_rp_engine/api/routes_rp.py`:
```python
@router.delete("/characters/{char_id}")
def delete_character(char_id: str, request: Request):
    store = request.app.state.store
    registry = getattr(request.app.state, "agent_registry", None)
    try:
        clean_id = _sanitize_key(char_id)
        path = os.path.join(store.char_dir, f"{clean_id}.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Character not found")
        os.remove(path)
        if registry:
            registry._rp_agents.pop(clean_id, None)
            registry._rp_runners.pop(clean_id, None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted", "char_id": clean_id}
```

3. Include `lorebook_router` in `src/story_rp_engine/api/app.py`:
```python
from story_rp_engine.api.routes_lorebook import router as lorebook_router
...
app.include_router(lorebook_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_lorebook_character_crud.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/api/routes_lorebook.py src/story_rp_engine/api/routes_rp.py src/story_rp_engine/api/app.py tests/test_api_lorebook_character_crud.py
git commit -m "feat(api): add lorebook CRUD and character deletion endpoints"
```

---

### Task 2: Backend Session History & Turn Deletion Endpoints

**Files:**
- Modify: `src/story_rp_engine/api/routes_rp.py`
- Test: `tests/test_api_session_turns.py`

**Interfaces:**
- Consumes: [`DatabaseSessionService`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/storage/store.py#L34-L36) (`get_session`, `delete_session`)
- Produces:
  - `GET /api/v1/rp/sessions/{session_id}/turns` -> `{"turns": List[dict]}`
  - `POST /api/v1/rp/sessions/{session_id}/turns/delete` -> `{"status": "ok", "remaining_turns": int}`
  - `DELETE /api/v1/rp/sessions/{session_id}` -> `{"status": "deleted", "session_id": str}`

- [ ] **Step 1: Write the failing test for session turns inspection and deletion**

Create `tests/test_api_session_turns.py`:
```python
import pytest
from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.config import EngineConfig


@pytest.fixture
def client(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path / "engine_data"))
    store = EngineStore(storage_dir=config.storage_dir)
    app = create_app(store=store, config=config)
    return TestClient(app)


def test_session_turns_management(client):
    session_id = "test_session_123"

    # Initially empty session or non-existent
    resp = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert resp.status_code == 200
    assert resp.json()["turns"] == []

    # Clear entire session
    del_resp = client.delete(f"/api/v1/rp/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "deleted", "session_id": session_id}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_session_turns.py -v`  
Expected: FAIL (404 Not Found).

- [ ] **Step 3: Implement session turns and deletion in `routes_rp.py`**

Add the session endpoints to `src/story_rp_engine/api/routes_rp.py`:
```python
from pydantic import BaseModel

class DeleteTurnRequest(BaseModel):
    turn_index: int
    truncate_subsequent: bool = False

@router.get("/rp/sessions/{session_id}/turns")
async def get_session_turns(session_id: str, request: Request):
    _sanitize_key(session_id)
    session_service = request.app.state.session_service
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id=session_id)
    if not session or not session.events:
        return {"turns": []}

    turns = []
    for idx, ev in enumerate(session.events):
        text = ""
        if ev.content and ev.content.parts:
            text = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
        role = getattr(ev.content, "role", "unknown") if ev.content else "system"
        if text:
            turns.append({"index": idx, "role": role, "text": text})
    return {"turns": turns}


@router.delete("/rp/sessions/{session_id}")
async def clear_session(session_id: str, request: Request):
    _sanitize_key(session_id)
    session_service = request.app.state.session_service
    await session_service.delete_session(app_name="rp_app", user_id="User", session_id=session_id)
    return {"status": "deleted", "session_id": session_id}


@router.post("/rp/sessions/{session_id}/turns/delete")
async def delete_session_turn(session_id: str, req: DeleteTurnRequest, request: Request):
    _sanitize_key(session_id)
    session_service = request.app.state.session_service
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id=session_id)
    if not session:
        return {"status": "ok", "remaining_turns": 0}

    # Recreate session with pruned events
    events = list(session.events)
    if req.truncate_subsequent:
        events = events[:req.turn_index]
    else:
        if 0 <= req.turn_index < len(events):
            events.pop(req.turn_index)

    await session_service.delete_session(app_name="rp_app", user_id="User", session_id=session_id)
    new_session = await session_service.create_session(app_name="rp_app", user_id="User", session_id=session_id, state=session.state)
    for ev in events:
        await session_service.append_event(new_session, ev)

    return {"status": "ok", "remaining_turns": len(events)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_session_turns.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/api/routes_rp.py tests/test_api_session_turns.py
git commit -m "feat(api): add session turn inspection and deletion endpoints"
```

---

### Task 3: Static Asset Serving Mount in FastAPI

**Files:**
- Modify: `src/story_rp_engine/api/app.py`
- Create: `src/story_rp_engine/web/index.html` (minimal initial shell)
- Test: `tests/test_web_mount.py`

**Interfaces:**
- Consumes: `fastapi.staticfiles.StaticFiles`, `pathlib.Path`
- Produces: `GET /` -> Serves `src/story_rp_engine/web/index.html`

- [ ] **Step 1: Write the failing test for static mount**

Create `tests/test_web_mount.py`:
```python
from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app

def test_web_ui_mounted():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Story & Roleplay Workbench" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web_mount.py -v`  
Expected: FAIL (404 Not Found at `/`).

- [ ] **Step 3: Create initial `web/index.html` and mount in `app.py`**

1. Create `src/story_rp_engine/web/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Story & Roleplay Workbench</title>
</head>
<body>
  <h1>Story & Roleplay Workbench</h1>
</body>
</html>
```

2. Modify `src/story_rp_engine/api/app.py`:
```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Inside create_app:
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web_ui")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_web_mount.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/api/app.py src/story_rp_engine/web/index.html tests/test_web_mount.py
git commit -m "feat(web): mount static frontend workbench to FastAPI root"
```

---

### Task 4: Complete Frontend Shell & Two-Column Layout

**Files:**
- Modify: `src/story_rp_engine/web/index.html`
- Create: `src/story_rp_engine/web/style.css`

**Interfaces:**
- Consumes: Tailwind CSS (CDN), Vue 3 (CDN), Lucide Icons (CDN), Marked.js (CDN)
- Produces: Responsive dark-mode shell with top navbar (tabs: Roleplay, Story Co-Pilot, Characters, Lorebooks) and responsive two-column grid containers.

- [ ] **Step 1: Write `src/story_rp_engine/web/style.css`**
Add clean dark-mode scrollbars, monospace font rules, and custom accent animations.

- [ ] **Step 2: Update `src/story_rp_engine/web/index.html`**
Include:
- CDN links for Tailwind CSS, Vue 3, Lucide, and Marked.js.
- `#app` container with Vue reactive bindings (`v-cloak`).
- Top Header: App Title, Navigation Tabs (`activeTab`), Backend Status Pill (`online`/`offline`).
- Tab 1: Roleplay Workbench container (Left parameters, Right chat bubbles + input).
- Tab 2: Story Co-Pilot container (Left parameters, Right document canvas + Director beats).
- Tab 3: Characters container (Left character list + import/export, Right full `CharacterCard` form).
- Tab 4: Lorebooks container (Left lorebook list + import/export, Right `Lorebook` & `LorebookEntry` editor).

- [ ] **Step 3: Verify with browser/curl check**
Run: `curl -s http://localhost:8000/ | grep "lucide"`  
Expected: HTML contains CDN links and Vue root element.

- [ ] **Step 4: Commit**

```bash
git add src/story_rp_engine/web/index.html src/story_rp_engine/web/style.css
git commit -m "feat(web): build workbench HTML shell and styling"
```

---

### Task 5: Frontend Vue 3 Logic — Characters & Lorebooks Workbench

**Files:**
- Create: `src/story_rp_engine/web/app.js`

**Interfaces:**
- Consumes:
  - `GET/POST/DELETE /api/v1/characters`
  - `GET/POST/DELETE /api/v1/lorebooks`
- Produces: Reactive character and lorebook management with import/export.

- [ ] **Step 1: Implement Character Card state and CRUD methods in `app.js`**
- State: `characters`, `selectedCharId`, `charForm` (mapping all fields: `char_id`, `name`, `description`, `personality`, `scenario`, `first_mes`, `alternate_greetings`, `mes_example`, `system_prompt`, `post_history_instructions`, `tags`, `creator_notes`).
- Methods: `loadCharacters()`, `selectCharacter(id)`, `newCharacter()`, `saveCharacter()`, `deleteCharacter()`, `importCharacterJSON(event)`, `exportCharacterJSON()`.

- [ ] **Step 2: Implement Lorebook state and CRUD methods in `app.js`**
- State: `lorebooks`, `selectedLorebookId`, `lorebookForm` (`name`, `description`, `entries: []`).
- Methods: `loadLorebooks()`, `selectLorebook(id)`, `newLorebook()`, `saveLorebook()`, `deleteLorebook()`, `addEntry()`, `removeEntry(idx)`, `importLorebookJSON(event)`, `exportLorebookJSON()`.

- [ ] **Step 3: Verify by loading characters and lorebooks via API test client**
Test saving and retrieving sample character and lorebook via frontend API calls.

- [ ] **Step 4: Commit**

```bash
git add src/story_rp_engine/web/app.js
git commit -m "feat(web): implement characters and lorebooks management logic"
```

---

### Task 6: Frontend Vue 3 Logic — Roleplay Workbench & Turn Deletion

**Files:**
- Modify: `src/story_rp_engine/web/app.js`

**Interfaces:**
- Consumes:
  - `POST /api/v1/rp/chat/stream`
  - `POST /api/v1/rp/sessions/{session_id}/turns/delete`
  - `DELETE /api/v1/rp/sessions/{session_id}`
- Produces: Real-time SSE streaming chat, message bubbles, turn actions, and conversation rewinding.

- [ ] **Step 1: Implement RP Chat State & Parameters in `app.js`**
- State: `rpSessionId`, `rpCharId`, `rpUserName`, `rpAuthorsNote`, `rpLorebookId`, `rpChunkSize`, `rpMessages`, `rpInput`, `isGeneratingRP`.

- [ ] **Step 2: Implement SSE Streaming `sendRPMessage()` with `AbortController`**
- Dispatches POST to `/api/v1/rp/chat/stream` with `RPChatRequest`.
- Reads SSE deltas and appends streaming tokens to active assistant bubble.
- Supports `stopGeneratingRP()`.

- [ ] **Step 3: Implement Turn Deletion & Rewind**
- `deleteTurn(index)`: Calls `/api/v1/rp/sessions/{session_id}/turns/delete` and removes bubble from `rpMessages`.
- `deleteFromHere(index)`: Truncates subsequent messages and rewinds the session.
- `regenerateTurn()`: Deletes last assistant message and re-sends.

- [ ] **Step 4: Commit**

```bash
git add src/story_rp_engine/web/app.js
git commit -m "feat(web): implement roleplay streaming chat and turn deletion"
```

---

### Task 7: Frontend Vue 3 Logic — Story Co-Pilot Workbench & Expansion Stream

**Files:**
- Modify: `src/story_rp_engine/web/app.js`

**Interfaces:**
- Consumes: `POST /api/v1/story/expand/stream`
- Produces: Document canvas with real-time SSE token append, word count, undo last expansion, and export.

- [ ] **Step 1: Implement Story Co-Pilot State & Parameters in `app.js`**
- State: `storySessionId`, `storyPremise`, `storyGenre`, `storyTone`, `storyInstruction`, `storyMaxTokens`, `storyChunkSize`, `storyCurrentText`, `previousStoryText`, `isGeneratingStory`, `directorBeats`.

- [ ] **Step 2: Implement SSE Streaming `expandStory()` with `AbortController`**
- Caches current draft to `previousStoryText` for undo support.
- Streams tokens from `/api/v1/story/expand/stream` and appends live to `storyCurrentText`.
- Supports `stopGeneratingStory()`.

- [ ] **Step 3: Implement Canvas Helpers**
- `undoLastExpansion()`: Restores `storyCurrentText` from `previousStoryText`.
- `exportStory(format)`: Generates downloadable `.txt` or `.md` file.
- Word count & estimated token calculations.

- [ ] **Step 4: Commit**

```bash
git add src/story_rp_engine/web/app.js
git commit -m "feat(web): implement story co-pilot expansion and canvas controls"
```

---

### Task 8: End-to-End System Integration & Verification

**Files:**
- Test: `tests/test_frontend_integration.py`

- [ ] **Step 1: Write integration tests covering all routes and static assets**
Create `tests/test_frontend_integration.py`:
- Test root route serves HTML with all 4 tab markers (`roleplay`, `story`, `characters`, `lorebooks`).
- Test static files `app.js` and `style.css` are accessible.
- Test character creation, listing, retrieval, and deletion.
- Test lorebook creation, listing, retrieval, and deletion.
- Test session turn endpoints.

- [ ] **Step 2: Run complete test suite**
Run: `uv run pytest -v`  
Expected: All tests pass (including existing and new tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_frontend_integration.py
git commit -m "test: add full integration suite for frontend workbench and API"
```
