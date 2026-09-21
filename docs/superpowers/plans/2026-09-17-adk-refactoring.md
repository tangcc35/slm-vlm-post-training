# Google ADK 2.0 Engine Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `story_rp_engine` to adhere strictly to Google ADK 2.0 patterns by eliminating per-turn agent instantiation, orchestrating story agents with declarative `Workflow` graphs, replacing ad-hoc prompt building with ADK callbacks, and executing runs natively via ADK `Runner` and `SessionService`.

**Architecture:** A long-lived `AgentRegistry` caches character agents and the story workflow graph singleton. Native ADK `Runner` and `InMemorySessionService` manage conversation turns, session state, and streaming. ADK `before_model_callback` dynamically injects lorebook world info and Author's Note steering directives without ad-hoc string concatenation or monkey-patching.

**Tech Stack:** Python 3.11, Google ADK 2.7.0 (`google-adk`), `google-genai`, `fastapi`, `pydantic>=2.0`, `pytest`.

**Spec:** [`docs/superpowers/specs/2026-09-17-adk-refactoring-design.md`](file:///home/tangc/slm-vlm-post-training/docs/superpowers/specs/2026-09-17-adk-refactoring-design.md)

## Global Constraints

- Python compatibility: 3.11 (`.venv`).
- Google ADK version: 2.7.0 (`google.adk`).
- No monkey-patching of ADK classes (`LlmAgent.invoke` / `LlmAgent.stream` must be completely removed).
- Agents and workflows must be instantiated once and cached/reused; never instantiated per-turn.
- Story Mode orchestration must use `google.adk.Workflow` with graph edges (`edges=[("START", director, writer)]`).
- Roleplay context injection must use ADK callbacks (`before_model_callback`) and dynamic instruction providers; no raw string concatenation of conversation history.
- REST and SSE streaming API JSON contracts must remain backward-compatible for all `/api/v1/*` endpoints.

---

### Task 1: ADK SessionService & Store Integration

**Files:**
- Modify: `src/story_rp_engine/storage/store.py`
- Test: `tests/story_rp_engine/test_storage.py`

**Interfaces:**
- Consumes: `google.adk.sessions.InMemorySessionService`
- Produces:
  - `EngineStore.session_service: InMemorySessionService`
  - `EngineStore.get_or_create_session(app_name: str, user_id: str, session_id: str, initial_state: Optional[dict] = None) -> Session`
  - Compatibility methods `save_history` and `get_history` delegating to ADK session events/state.

- [ ] **Step 1: Write failing tests for SessionService integration in test_storage.py**

```python
# Append to tests/story_rp_engine/test_storage.py
import pytest
from story_rp_engine.storage.store import EngineStore
from google.adk.sessions import Session

@pytest.mark.anyio
async def test_store_session_service_lifecycle(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    session = await store.get_or_create_session(
        app_name="rp_app",
        user_id="user_123",
        session_id="session_abc",
        initial_state={"char_id": "lyra"}
    )
    assert isinstance(session, Session)
    assert session.id == "session_abc"
    assert session.state.get("char_id") == "lyra"

    # Re-fetching returns the existing session
    session_again = await store.get_or_create_session(
        app_name="rp_app",
        user_id="user_123",
        session_id="session_abc"
    )
    assert session_again.id == "session_abc"
    assert session_again.state.get("char_id") == "lyra"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/story_rp_engine/test_storage.py::test_store_session_service_lifecycle -v`  
Expected: FAIL with `AttributeError: 'EngineStore' object has no attribute 'get_or_create_session'`

- [ ] **Step 3: Implement get_or_create_session and session_service in store.py**

```python
# In src/story_rp_engine/storage/store.py
from typing import Dict, List, Optional, Any
from google.adk.sessions import InMemorySessionService, Session

class EngineStore:
    def __init__(self, storage_dir: str = ".engine_data", session_service: Optional[InMemorySessionService] = None):
        self.storage_dir = storage_dir
        self.char_dir = os.path.join(storage_dir, "characters")
        self.lorebooks_dir = os.path.join(storage_dir, "lorebooks")
        self.sessions_dir = os.path.join(storage_dir, "sessions")
        os.makedirs(self.char_dir, exist_ok=True)
        os.makedirs(self.lorebooks_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.session_service = session_service or InMemorySessionService()

    async def get_or_create_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Session:
        session_id = _sanitize_key(session_id)
        user_id = _sanitize_key(user_id)
        session = await self.session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        if session is None:
            session = await self.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {},
            )
        return session
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/story_rp_engine/test_storage.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/storage/store.py tests/story_rp_engine/test_storage.py
git commit -m "feat: integrate ADK InMemorySessionService into EngineStore"
```

---

### Task 2: ADK Callbacks for RP Context (Lorebook & Steering Injection)

**Files:**
- Create: `src/story_rp_engine/rp/callbacks.py`
- Modify: `src/story_rp_engine/rp/prompt_builder.py`
- Create: `tests/story_rp_engine/test_rp_callbacks.py`
- Modify: `tests/story_rp_engine/test_prompt_builder.py`

**Interfaces:**
- Consumes:
  - `google.adk.agents.callback_context.CallbackContext`
  - `google.adk.models.LlmRequest`
  - `story_rp_engine.rp.lorebook.LorebookEngine`
  - `story_rp_engine.core.types.Lorebook`
- Produces:
  - `create_rp_before_model_callback(lorebook: Optional[Lorebook] = None) -> Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]`
  - `build_character_system_instruction(card: CharacterCardV2, user_name: str = "User") -> str`

- [ ] **Step 1: Write failing test for RP callbacks in test_rp_callbacks.py**

```python
# tests/story_rp_engine/test_rp_callbacks.py
import pytest
from unittest.mock import MagicMock
from google.genai import types
from google.adk.models import LlmRequest
from google.adk.agents.callback_context import CallbackContext
from story_rp_engine.core.types import Lorebook, LorebookEntry
from story_rp_engine.rp.callbacks import create_rp_before_model_callback

def test_before_model_callback_injects_lore_and_authors_note():
    lore = Lorebook(
        name="Fantasy World",
        entries=[
            LorebookEntry(keys=["sword"], content="Excalibur is a legendary blade."),
        ],
    )
    cb = create_rp_before_model_callback(lorebook=lore)

    mock_context = MagicMock(spec=CallbackContext)
    mock_context.state = {"authors_note": "Tone: mysterious"}
    
    request = LlmRequest(
        model="test-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="I draw my silver sword!")]
            )
        ],
        config=types.GenerateContentConfig(system_instruction="Base character prompt.")
    )

    res = cb(mock_context, request)
    assert res is None  # Allow generation to proceed
    
    # Verify lore and author's note were injected into system_instruction or dynamic instruction
    instruction = str(request.config.system_instruction)
    assert "Excalibur is a legendary blade." in instruction
    assert "Tone: mysterious" in instruction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/story_rp_engine/test_rp_callbacks.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'story_rp_engine.rp.callbacks'`

- [ ] **Step 3: Implement callbacks in src/story_rp_engine/rp/callbacks.py**

```python
# src/story_rp_engine/rp/callbacks.py
from typing import Callable, Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from story_rp_engine.core.types import Lorebook
from story_rp_engine.rp.lorebook import LorebookEngine


def create_rp_before_model_callback(
    lorebook: Optional[Lorebook] = None,
) -> Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]:
    """Creates an ADK before_model_callback that injects active lore and author's note."""
    engine = LorebookEngine(lorebook) if lorebook else None

    def rp_before_model_callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        # Extract user input from latest user content
        user_text = ""
        if llm_request.contents:
            for content in reversed(llm_request.contents):
                if content.role == "user" and content.parts:
                    user_text = " ".join([p.text for p in content.parts if getattr(p, "text", None)])
                    break

        extra_sections = []

        # 1. Match Lorebook entries
        if engine and user_text:
            active_lore = engine.match_entries(user_text)
            if active_lore:
                lore_text = "\n".join([f"- {entry.content}" for entry in active_lore])
                extra_sections.append(f"### Relevant World Information\n{lore_text}")

        # 2. Author's note from session state
        authors_note = callback_context.state.get("authors_note")
        if authors_note and str(authors_note).strip():
            extra_sections.append(f"### Narrative Directive\n{str(authors_note).strip()}")

        if extra_sections:
            additions = "\n\n".join(extra_sections)
            current_instruction = ""
            if llm_request.config and llm_request.config.system_instruction:
                inst = llm_request.config.system_instruction
                current_instruction = inst if isinstance(inst, str) else str(inst)
            
            new_instruction = f"{current_instruction}\n\n{additions}".strip()
            if not llm_request.config:
                llm_request.config = types.GenerateContentConfig()
            llm_request.config.system_instruction = new_instruction

        return None

    return rp_before_model_callback
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/story_rp_engine/test_rp_callbacks.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/rp/callbacks.py tests/story_rp_engine/test_rp_callbacks.py
git commit -m "feat: implement ADK before_model_callback for dynamic lore and steering injection"
```

---

### Task 3: Agent Registry & Singleton Lifecycle Management

**Files:**
- Create: `src/story_rp_engine/core/agent_registry.py`
- Modify: `src/story_rp_engine/rp/agent.py`
- Create: `tests/story_rp_engine/test_agent_registry.py`
- Modify: `tests/story_rp_engine/test_rp_agent.py`

**Interfaces:**
- Consumes:
  - `story_rp_engine.core.config.EngineConfig`
  - `story_rp_engine.storage.store.EngineStore`
  - `story_rp_engine.core.types.CharacterCardV2`
  - `story_rp_engine.rp.callbacks.create_rp_before_model_callback`
- Produces:
  - `AgentRegistry(config: EngineConfig, store: EngineStore)`
  - `AgentRegistry.get_or_create_rp_agent(char_id: str, card: Optional[CharacterCardV2] = None) -> LlmAgent` (created once per character, cached)
  - `AgentRegistry.get_story_workflow() -> Workflow` (created once, cached)
  - `create_rp_agent(card: CharacterCardV2, config: EngineConfig, lorebook: Optional[Lorebook] = None, user_name: str = "User") -> LlmAgent`

- [ ] **Step 1: Write failing test for AgentRegistry in test_agent_registry.py**

```python
# tests/story_rp_engine/test_agent_registry.py
import pytest
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data

def test_agent_registry_instantiates_agent_only_once(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Seraphina",
            description="High Priestess",
            personality="Serene",
            scenario="Temple",
            first_mes="Blessings upon you.",
            mes_example="",
        )
    )
    store.save_character("seraphina", card)

    # First access creates agent
    agent1 = registry.get_or_create_rp_agent("seraphina")
    assert agent1.name == "rp_seraphina"

    # Second access returns the exact same cached instance
    agent2 = registry.get_or_create_rp_agent("seraphina")
    assert agent1 is agent2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/story_rp_engine/test_agent_registry.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'story_rp_engine.core.agent_registry'`

- [ ] **Step 3: Implement AgentRegistry and update rp/agent.py**

```python
# src/story_rp_engine/rp/agent.py
import re
from typing import Optional
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.core.types import CharacterCardV2, Lorebook
from story_rp_engine.rp.callbacks import create_rp_before_model_callback
from story_rp_engine.rp.prompt_builder import build_rp_system_instruction


def create_rp_agent(
    card: CharacterCardV2,
    config: EngineConfig,
    lorebook: Optional[Lorebook] = None,
    user_name: str = "User",
) -> LlmAgent:
    """Creates a Google ADK LlmAgent configured for character roleplay."""
    model = get_adk_model(config)
    instruction = build_rp_system_instruction(card, active_lore=None, user_name=user_name)
    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", card.data.name.lower()).strip("_")
    before_cb = create_rp_before_model_callback(lorebook=lorebook)

    return LlmAgent(
        name=f"rp_{sanitized_name}",
        model=model,
        instruction=instruction,
        before_model_callback=before_cb,
    )
```

```python
# src/story_rp_engine/core/agent_registry.py
from typing import Dict, Optional
from google.adk.agents import LlmAgent
from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.types import CharacterCardV2, Lorebook
from story_rp_engine.rp.agent import create_rp_agent


class AgentRegistry:
    """Central registry ensuring agents and workflows are created once and reused."""

    def __init__(self, config: EngineConfig, store: EngineStore):
        self.config = config
        self.store = store
        self._rp_agents: Dict[str, LlmAgent] = {}
        self._story_workflow: Optional[Workflow] = None

    def get_or_create_rp_agent(
        self,
        char_id: str,
        card: Optional[CharacterCardV2] = None,
        lorebook: Optional[Lorebook] = None,
    ) -> LlmAgent:
        if char_id in self._rp_agents:
            return self._rp_agents[char_id]

        if card is None:
            card = self.store.get_character(char_id)
        if not card:
            raise ValueError(f"Character {char_id} not found")

        agent = create_rp_agent(card, self.config, lorebook=lorebook)
        self._rp_agents[char_id] = agent
        return agent

    def register_rp_agent(self, char_id: str, agent: LlmAgent) -> None:
        self._rp_agents[char_id] = agent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/story_rp_engine/test_agent_registry.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/rp/agent.py src/story_rp_engine/core/agent_registry.py tests/story_rp_engine/test_agent_registry.py
git commit -m "feat: implement AgentRegistry for singleton agent lifecycle management"
```

---

### Task 4: Story Mode Graph Workflow (`google.adk.Workflow`)

**Files:**
- Modify: `src/story_rp_engine/story/director_agent.py`
- Modify: `src/story_rp_engine/story/writer_agent.py`
- Modify: `src/story_rp_engine/story/workflow.py`
- Modify: `tests/story_rp_engine/test_story_workflow.py`

**Interfaces:**
- Consumes:
  - `google.adk.Workflow`
  - `google.adk.agents.LlmAgent`
  - `story_rp_engine.core.config.EngineConfig`
- Produces:
  - `create_story_workflow(config: EngineConfig) -> Workflow` with `edges=[("START", director_agent, writer_agent)]`
  - `format_story_director_input(request: StoryRequest) -> str`

- [ ] **Step 1: Write failing test for Story Graph Workflow in test_story_workflow.py**

```python
# In tests/story_rp_engine/test_story_workflow.py
from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.story.workflow import create_story_workflow

def test_create_story_workflow_graph_structure():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    wf = create_story_workflow(config)

    assert isinstance(wf, Workflow)
    assert wf.name == "story_workflow"
    assert len(wf.graph.nodes) == 3  # __START__, story_director, story_writer
    node_names = [n.name for n in wf.graph.nodes]
    assert "story_director" in node_names
    assert "story_writer" in node_names
    
    # Verify graph edge sequence: START -> director -> writer
    edge_pairs = [(e.from_node.name, e.to_node.name) for e in wf.graph.edges]
    assert ("__START__", "story_director") in edge_pairs
    assert ("story_director", "story_writer") in edge_pairs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/story_rp_engine/test_story_workflow.py::test_create_story_workflow_graph_structure -v`  
Expected: FAIL with `ImportError: cannot import name 'create_story_workflow' from 'story_rp_engine.story.workflow'`

- [ ] **Step 3: Implement create_story_workflow using ADK Workflow**

```python
# In src/story_rp_engine/story/workflow.py
from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent


def format_story_input(request: StoryRequest) -> str:
    """Formats the initial story request into the director node's input prompt."""
    return (
        f"Premise: {request.premise or 'Not specified'}\n"
        f"Genre: {request.genre}\n"
        f"Tone: {request.tone}\n"
        f"Current Text:\n{request.current_text}\n\n"
        f"User Instruction: {request.instruction or 'Continue the story.'}\n"
        "Provide brief scene framing and narrative guidance for the writer."
    )


def create_story_workflow(config: EngineConfig) -> Workflow:
    """Creates a declarative ADK Graph Workflow connecting Director and Writer agents."""
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    return Workflow(
        name="story_workflow",
        edges=[
            ("START", director, writer)
        ],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/story_rp_engine/test_story_workflow.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/story/workflow.py tests/story_rp_engine/test_story_workflow.py
git commit -m "feat: implement declarative ADK Workflow graph for story mode"
```

---

### Task 5: Native ADK Runner Execution, Clean API Routes & Remove Monkey-Patches

**Files:**
- Modify: `src/story_rp_engine/core/agent_utils.py`
- Modify: `src/story_rp_engine/api/app.py`
- Modify: `src/story_rp_engine/api/routes_rp.py`
- Modify: `src/story_rp_engine/api/routes_story.py`
- Test: `tests/story_rp_engine/test_api.py`
- Test: `tests/story_rp_engine/test_e2e_integration.py`

**Interfaces:**
- Consumes:
  - `google.adk.Runner`
  - `google.genai.types`
  - `AgentRegistry`
- Produces:
  - `execute_runner_turn(runner: Runner, user_id: str, session_id: str, message: str, state_delta: Optional[dict] = None) -> str`
  - `stream_runner_turn(runner: Runner, user_id: str, session_id: str, message: str, state_delta: Optional[dict] = None) -> AsyncIterator[str]`
  - Async FastAPI endpoints `/api/v1/rp/chat`, `/stream`, `/api/v1/story/expand`, `/stream`.

- [ ] **Step 1: Write failing test in test_api.py for async Runner execution without monkey patch**

```python
# In tests/story_rp_engine/test_api.py
@pytest.mark.anyio
async def test_native_runner_chat_execution(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    
    # Pre-register character
    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Lyra", description="A", personality="B", scenario="C", first_mes="D", mes_example=""
        )
    )
    store.save_character("lyra", card)
    
    # Verify agent is created once in registry and app.state
    assert hasattr(app.state, "agent_registry")
    assert hasattr(app.state, "runner")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/story_rp_engine/test_api.py::test_native_runner_chat_execution -v`  
Expected: FAIL with `AttributeError: 'State' object has no attribute 'agent_registry'`

- [ ] **Step 3: Clean agent_utils.py, update app.py and route handlers**

In `src/story_rp_engine/core/agent_utils.py`:
- Remove monkey-patching of `LlmAgent.invoke` and `LlmAgent.stream`.
- Add clean async helpers `execute_runner_turn` and `stream_runner_turn` using `Runner.run_async()`.

In `src/story_rp_engine/api/app.py`:
- Initialize `AgentRegistry(config, store)` and store on `app.state.agent_registry`.
- Initialize `Runner` instances for RP and Story workflows.

In `src/story_rp_engine/api/routes_rp.py`:
- Change endpoints to `async def`.
- Look up character agent from `app.state.agent_registry.get_or_create_rp_agent(req.char_id)`.
- Execute turn via `runner.run_async()`.

In `src/story_rp_engine/api/routes_story.py`:
- Change endpoints to `async def`.
- Run `app.state.agent_registry.get_story_workflow()` via `runner.run_async()`.

- [ ] **Step 4: Run full test suite to verify all tests pass**

Run: `pytest tests/story_rp_engine -v`  
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/core/agent_utils.py src/story_rp_engine/api/app.py src/story_rp_engine/api/routes_rp.py src/story_rp_engine/api/routes_story.py tests/story_rp_engine/
git commit -m "refactor: eliminate monkey-patching and execute turns natively via ADK Runner"
```

---

## Execution Choice

Two execution options are available:
1. **Subagent-Driven Development (recommended)**: Dispatch a fresh subagent per task, verify each change, and checkpoint progress.
2. **Inline Execution**: Execute tasks directly in this session with review checkpoints.
