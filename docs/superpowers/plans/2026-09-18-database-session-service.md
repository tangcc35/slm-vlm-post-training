# DatabaseSessionService Implementation & Store History Deprecation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Google ADK's native `DatabaseSessionService` as the default session storage in `EngineStore`, and remove legacy `store.save_history` and `store.get_history` file-based JSON persistence.

**Architecture:** Replace `InMemorySessionService` with `DatabaseSessionService` (backed by SQLite via `aiosqlite`) in `EngineStore`. The ADK `Runner` automatically writes all conversational events (user inputs and assistant replies) to the database during turn execution. Remove redundant manual JSON history writes and reads from `routes_rp.py`, updating API integration tests to assert against ADK `session_service` directly.

**Tech Stack:** Python 3.11, Google ADK 2.7.0 (`DatabaseSessionService`), SQLAlchemy 2.0 (`aiosqlite`), FastAPI, Pytest.

**Spec:** Requirement from user: Implement `DatabaseSessionService`, and remove `store.save_history` and `store.get_history` related code.

## Global Constraints

- Never break existing character or lorebook storage in `EngineStore`.
- All RP and Story runner turns must execute cleanly through ADK's `Runner.run_async()`.
- Default to SQLite file database in `.engine_data/sessions.db` (or configurable via `db_url`).
- Remove all filesystem JSON session history code in `.engine_data/sessions/`.
- Maintain 100% passing test suite across `tests/story_rp_engine/`.

---

### Task 1: Add Database Dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:24-32`

**Interfaces:**
- Produces: Project dependencies updated with `sqlalchemy>=2.0.0` and `aiosqlite>=0.20.0`.

- [ ] **Step 1: Update pyproject.toml dependencies**

Add `"sqlalchemy>=2.0.0"` and `"aiosqlite>=0.20.0"` to the `dependencies` list in `pyproject.toml`.

- [ ] **Step 2: Verify package installation**

Run: `python3 -c "import sqlalchemy; import aiosqlite; from google.adk.sessions import DatabaseSessionService; print('OK')"`  
Expected: Prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add sqlalchemy and aiosqlite for ADK DatabaseSessionService"
```

---

### Task 2: Refactor `EngineStore` with `DatabaseSessionService` & Remove History Methods

**Files:**
- Modify: `src/story_rp_engine/storage/store.py`
- Modify: `tests/story_rp_engine/test_storage.py`

**Interfaces:**
- Consumes:
  - `google.adk.sessions.DatabaseSessionService`
  - `google.adk.sessions.BaseSessionService`
- Produces:
  - `EngineStore(storage_dir: str = ".engine_data", session_service: Optional[BaseSessionService] = None, db_url: Optional[str] = None)`
  - Removal of `EngineStore.save_history` and `EngineStore.get_history`
  - Removal of `EngineStore.sessions_dir`

- [ ] **Step 1: Write tests for DatabaseSessionService and remove old history tests in test_storage.py**

In `tests/story_rp_engine/test_storage.py`:
- Remove `test_session_history`, `test_session_history_empty_when_missing`, and `test_session_history_append`.
- In `test_path_traversal_defense`, remove calls to `store.save_history` and `store.get_history`.
- Add test `test_database_session_service_default(tmp_path)`:
  ```python
  @pytest.mark.anyio
  async def test_database_session_service_default(tmp_path):
      store = EngineStore(storage_dir=str(tmp_path))
      assert isinstance(store.session_service, DatabaseSessionService)
      session = await store.get_or_create_session(
          app_name="test_app", user_id="User", session_id="sess_1"
      )
      assert session.id == "sess_1"
      fetched = await store.session_service.get_session(
          app_name="test_app", user_id="User", session_id="sess_1"
      )
      assert fetched is not None
      assert fetched.id == "sess_1"
  ```
- Add test `test_database_session_service_custom_url(tmp_path)`:
  ```python
  @pytest.mark.anyio
  async def test_database_session_service_custom_url(tmp_path):
      db_file = tmp_path / "custom.db"
      store = EngineStore(
          storage_dir=str(tmp_path),
          db_url=f"sqlite+aiosqlite:///{db_file}",
      )
      assert isinstance(store.session_service, DatabaseSessionService)
      session = await store.get_or_create_session(
          app_name="test_app", user_id="User", session_id="sess_custom"
      )
      assert session.id == "sess_custom"
  ```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/story_rp_engine/test_storage.py -v`  
Expected: FAIL with `AssertionError: assert isinstance(InMemorySessionService, DatabaseSessionService)`.

- [ ] **Step 3: Implement DatabaseSessionService and remove history methods in store.py**

In `src/story_rp_engine/storage/store.py`:
- Import `BaseSessionService`, `DatabaseSessionService`, `Session` from `google.adk.sessions`.
- In `EngineStore.__init__`:
  ```python
  def __init__(
      self,
      storage_dir: str = ".engine_data",
      session_service: Optional[BaseSessionService] = None,
      db_url: Optional[str] = None,
  ):
      self.storage_dir = storage_dir
      self.char_dir = os.path.join(storage_dir, "characters")
      self.lorebooks_dir = os.path.join(storage_dir, "lorebooks")
      os.makedirs(self.char_dir, exist_ok=True)
      os.makedirs(self.lorebooks_dir, exist_ok=True)

      if session_service is not None:
          self.session_service = session_service
      else:
          resolved_db_url = db_url or f"sqlite+aiosqlite:///{os.path.abspath(os.path.join(storage_dir, 'sessions.db'))}"
          self.session_service = DatabaseSessionService(db_url=resolved_db_url)
  ```
- Delete `self.sessions_dir`.
- Delete `save_history` and `get_history` methods.
- Remove `ChatMessage` from imports in `store.py` if no longer used there.

- [ ] **Step 4: Run storage tests to verify they pass**

Run: `pytest tests/story_rp_engine/test_storage.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/storage/store.py tests/story_rp_engine/test_storage.py
git commit -m "refactor: use DatabaseSessionService in EngineStore and remove save/get_history"
```

---

### Task 3: Remove Redundant History Calls from `routes_rp.py`

**Files:**
- Modify: `src/story_rp_engine/api/routes_rp.py:51-125`

**Interfaces:**
- Consumes:
  - `execute_runner_turn(runner, user_id, session_id, message, state_delta)`
  - `stream_runner_turn(runner, user_id, session_id, message, state_delta)`
- Produces:
  - Cleaned `chat_rp` and `chat_rp_stream` handlers without `store.get_history` or `store.save_history`.

- [ ] **Step 1: Clean chat_rp in routes_rp.py**

In `chat_rp`:
- Remove `history = store.get_history(req.session_id)`.
- Remove `updated_history = history + [...]` and `store.save_history(...)`.
- Handler only checks character existence, gets runner, calls `execute_runner_turn`, and returns `{"reply": reply, "session_id": req.session_id}`.

- [ ] **Step 2: Clean chat_rp_stream in routes_rp.py**

In `chat_rp_stream`:
- Remove `history = store.get_history(req.session_id)`.
- Remove `accumulated_chunks` and the `finally:` block that called `store.save_history`.
- Generator simply yields chunks and returns `StreamingResponse(event_stream(), media_type="text/event-stream")`.
- Remove unused `ChatMessage` import.

- [ ] **Step 3: Commit**

```bash
git add src/story_rp_engine/api/routes_rp.py
git commit -m "refactor: remove manual history saving from routes_rp handlers"
```

---

### Task 4: Update API & Integration Tests

**Files:**
- Modify: `tests/story_rp_engine/test_api.py`
- Modify: `tests/story_rp_engine/test_e2e_integration.py`

**Interfaces:**
- Asserts: `session_service.get_session(...)` directly to verify ADK multi-turn persistence.

- [ ] **Step 1: Update test_api.py**

- In `test_rp_chat_endpoint`: remove `history = store.get_history("session_1")` checks since runner turn was mocked.
- In `test_rp_chat_stream_endpoint`: remove `history = store.get_history("session_2")` checks.
- In `test_native_runner_chat_execution`:
  - Replace `store.get_history("session_native")` with:
    ```python
    session = await app.state.session_service.get_session(
        app_name="rp_app", user_id="User", session_id="session_native"
    )
    assert session is not None
    assert len(session.events) >= 2
    user_event = session.events[0]
    model_event = session.events[1]
    assert user_event.content.parts[0].text == "Sing for me."
    assert model_event.content.parts[0].text == "I sing a ballad."
    ```
- In `test_native_runner_stream_execution`:
  - Replace `store.get_history("session_native_stream")` with:
    ```python
    session = await app.state.session_service.get_session(
        app_name="rp_app", user_id="User", session_id="session_native_stream"
    )
    assert session is not None
    assert len(session.events) >= 2
    ```

- [ ] **Step 2: Update test_e2e_integration.py**

In `tests/story_rp_engine/test_e2e_integration.py`:
- Remove step 4 (`saved_history = store.get_history("sess_mountain_1")`), or update it to verify that characters and story endpoints work in conjunction without `get_history`.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/story_rp_engine -v`  
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/story_rp_engine/test_api.py tests/story_rp_engine/test_e2e_integration.py
git commit -m "test: update API and integration tests to verify DatabaseSessionService events"
```
