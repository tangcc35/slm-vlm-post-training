# Dual-Mode Story & Roleplay Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular dual-mode AI engine for Character Roleplay and General Story Co-Pilot using Google ADK (`google-adk`), LiteLLM, and FastAPI with local model support.

**Architecture:** Decoupled Python service with Character Card V2 and prompt assembly for RP, ADK Director & Writer multi-agent workflow for Story, unified under a FastAPI REST and Server-Sent Events (SSE) streaming service.

**Tech Stack:** Python 3.11, Google ADK (`google-adk`), `litellm`, `fastapi`, `uvicorn`, `pydantic>=2.0`, `pytest`, `httpx`.

**Spec:** [`docs/superpowers/specs/2026-09-16-story-rp-engine-design.md`](file:///home/tangc/slm-vlm-post-training/docs/superpowers/specs/2026-09-16-story-rp-engine-design.md)

## Global Constraints

- Python compatibility: `>=3.10,<3.12` (managed via `uv`).
- Local-first inference: Primary target is Ollama (`http://localhost:11434`) and OpenAI-compatible endpoints (`http://localhost:8000/v1`), model-agnostic via `LiteLlm`.
- Character Card compliance: Character Card V2 specification (`chara_card_v2`, version `2.0`).
- Multi-agent orchestration: Google ADK `LlmAgent` and workflow runners.
- No UI code in this phase: Backend-first; REST API and SSE streaming only.

---

### Task 1: Project Dependencies, Core Config, and Data Schemas

**Files:**
- Modify: `pyproject.toml`
- Create: `src/story_rp_engine/__init__.py`
- Create: `src/story_rp_engine/core/__init__.py`
- Create: `src/story_rp_engine/core/config.py`
- Create: `src/story_rp_engine/core/types.py`
- Test: `tests/story_rp_engine/test_types.py`

**Interfaces:**
- Consumes: None (root data foundation).
- Produces:
  - `EngineConfig(model_name: str, api_base: Optional[str], api_key: Optional[str], temperature: float, top_p: float, max_tokens: int)`
  - `CharacterCardV2(spec: str, spec_version: str, data: CharacterCardV2Data)`
  - `CharacterCardV2Data(...)`
  - `LorebookEntry(keys: List[str], content: str, insertion_order: int, enabled: bool)`
  - `Lorebook(name: str, description: Optional[str], entries: List[LorebookEntry])`
  - `ChatMessage(role: Literal["user", "assistant", "system"], content: str, timestamp: datetime)`
  - `StoryRequest(premise: Optional[str], current_text: str, instruction: Optional[str], genre: Optional[str], tone: Optional[str], max_tokens: int)`

- [ ] **Step 1: Write failing test for schemas and config**

```python
# tests/story_rp_engine/test_types.py
import pytest
from pydantic import ValidationError
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import (
    CharacterCardV2,
    CharacterCardV2Data,
    Lorebook,
    LorebookEntry,
    ChatMessage,
    StoryRequest,
)

def test_engine_config_defaults():
    config = EngineConfig()
    assert config.model_name == "ollama/llama3.1:8b"
    assert config.temperature == 0.8
    assert config.max_tokens == 512

def test_character_card_v2_valid():
    data = CharacterCardV2Data(
        name="Seraphina",
        description="A quiet archivist",
        personality="Cautious, observant",
        scenario="An ancient library after hours",
        first_mes="Welcome to the archives. Please keep your voice down.",
        mes_example="<START>\n{{user}}: What is this place?\n{{char}}: It is memory preserved.",
    )
    card = CharacterCardV2(data=data)
    assert card.spec == "chara_card_v2"
    assert card.spec_version == "2.0"
    assert card.data.name == "Seraphina"

def test_character_card_v2_invalid_spec():
    data = CharacterCardV2Data(
        name="A", description="B", personality="C", scenario="D", first_mes="E", mes_example="F"
    )
    with pytest.raises(ValidationError):
        CharacterCardV2(spec="invalid_spec", data=data)

def test_lorebook_and_entry():
    entry = LorebookEntry(keys=["archive", "library"], content="The Archives were founded in 1420.")
    lorebook = Lorebook(name="Setting Lore", entries=[entry])
    assert len(lorebook.entries) == 1
    assert "archive" in lorebook.entries[0].keys

def test_story_request_defaults():
    req = StoryRequest(current_text="The wind howled.")
    assert req.current_text == "The wind howled."
    assert req.genre == "Fiction"
    assert req.tone == "Balanced"
    assert req.max_tokens == 512
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'story_rp_engine'`

- [ ] **Step 3: Add dependencies to pyproject.toml and implement config & types**

Add packages:
```bash
uv add google-adk litellm fastapi uvicorn pydantic
```

Update `pyproject.toml` wheel packages list to include `src/story_rp_engine`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/slm_post_train", "src/story_rp_engine"]
```

Implement `src/story_rp_engine/core/config.py`:
```python
from typing import Optional
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    model_name: str = Field(default="ollama/llama3.1:8b", description="Model identifier for LiteLLM")
    api_base: Optional[str] = Field(default=None, description="Custom endpoint URL e.g. http://localhost:11434")
    api_key: Optional[str] = Field(default=None, description="API key if required (e.g. for Gemini/OpenAI)")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
```

Implement `src/story_rp_engine/core/types.py`:
```python
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class CharacterCardV2Data(BaseModel):
    name: str
    description: str
    personality: str
    scenario: str
    first_mes: str
    mes_example: str
    system_prompt: Optional[str] = None
    post_history_instructions: Optional[str] = None
    alternate_greetings: List[str] = Field(default_factory=list)
    creator_notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class CharacterCardV2(BaseModel):
    spec: Literal["chara_card_v2"] = "chara_card_v2"
    spec_version: Literal["2.0"] = "2.0"
    data: CharacterCardV2Data

class LorebookEntry(BaseModel):
    keys: List[str]
    content: str
    insertion_order: int = 100
    enabled: bool = True

class Lorebook(BaseModel):
    name: str
    description: Optional[str] = None
    entries: List[LorebookEntry] = Field(default_factory=list)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StoryRequest(BaseModel):
    premise: Optional[str] = None
    current_text: str = ""
    instruction: Optional[str] = "Continue the story naturally from the current point."
    genre: Optional[str] = "Fiction"
    tone: Optional[str] = "Balanced"
    max_tokens: int = Field(default=512, ge=1, le=4096)
```

Create empty `__init__.py` files for packages.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/story_rp_engine/test_types.py -v`
Expected: PASS (4 tests passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/story_rp_engine/ tests/story_rp_engine/
git commit -m "feat(core): add engine configuration and Pydantic types"
```

---

### Task 2: Model Provider Factory (Google ADK & LiteLLM Integration)

**Files:**
- Create: `src/story_rp_engine/core/model_provider.py`
- Test: `tests/story_rp_engine/test_model_provider.py`

**Interfaces:**
- Consumes: `EngineConfig` from `src/story_rp_engine/core/config.py`
- Produces: `get_adk_model(config: EngineConfig) -> LiteLlm`

- [ ] **Step 1: Write failing test for model provider**

```python
# tests/story_rp_engine/test_model_provider.py
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from google.adk.models.lite_llm import LiteLlm

def test_get_adk_model_ollama():
    config = EngineConfig(model_name="ollama/llama3.1:8b", api_base="http://localhost:11434")
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert model.model == "ollama/llama3.1:8b" or model.model_name == "ollama/llama3.1:8b"

def test_get_adk_model_openai_compat():
    config = EngineConfig(
        model_name="openai/gemma-2-9b-it",
        api_base="http://localhost:8000/v1",
        api_key="custom-key"
    )
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_model_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'story_rp_engine.core.model_provider'`

- [ ] **Step 3: Implement model provider factory**

```python
# src/story_rp_engine/core/model_provider.py
import os
from google.adk.models.lite_llm import LiteLlm
from story_rp_engine.core.config import EngineConfig

def get_adk_model(config: EngineConfig) -> LiteLlm:
    """Instantiates and configures a Google ADK LiteLlm model wrapper."""
    if config.api_base:
        os.environ["LITELLM_API_BASE"] = config.api_base
    if config.api_key:
        os.environ["OPENAI_API_KEY"] = config.api_key

    # LiteLlm accepts model_name and passes through keyword options
    kwargs = {}
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_key:
        kwargs["api_key"] = config.api_key

    try:
        return LiteLlm(model_name=config.model_name, **kwargs)
    except TypeError:
        # Fallback for alternative parameter signature
        return LiteLlm(model=config.model_name, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/story_rp_engine/test_model_provider.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/core/model_provider.py tests/story_rp_engine/test_model_provider.py
git commit -m "feat(core): add Google ADK LiteLlm model provider factory"
```

---

### Task 3: Character Card Parser & Lorebook Engine

**Files:**
- Create: `src/story_rp_engine/rp/__init__.py`
- Create: `src/story_rp_engine/rp/character.py`
- Create: `src/story_rp_engine/rp/lorebook.py`
- Test: `tests/story_rp_engine/test_character.py`
- Test: `tests/story_rp_engine/test_lorebook.py`

**Interfaces:**
- Consumes: `CharacterCardV2`, `CharacterCardV2Data`, `Lorebook`, `LorebookEntry` from `core/types.py`
- Produces:
  - `load_character_from_dict(payload: dict) -> CharacterCardV2`
  - `load_character_from_json(json_str: str) -> CharacterCardV2`
  - `LorebookEngine.find_matching_entries(lorebooks: List[Lorebook], text: str) -> List[LorebookEntry]`

- [ ] **Step 1: Write failing tests for character parser and lorebook engine**

```python
# tests/story_rp_engine/test_character.py
import json
import pytest
from story_rp_engine.rp.character import load_character_from_dict, load_character_from_json

def test_load_character_from_valid_v2_dict():
    payload = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Kaelen",
            "description": "Ranger of the North",
            "personality": "Stoic, loyal",
            "scenario": "A campfire in the woods",
            "first_mes": "Keep your voice down. Something is out there.",
            "mes_example": "{{user}}: What was that?\n{{char}}: Wolves, likely.",
        }
    }
    card = load_character_from_dict(payload)
    assert card.data.name == "Kaelen"

def test_load_character_from_json_string():
    json_str = json.dumps({
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Elysia",
            "description": "Mage",
            "personality": "Curious",
            "scenario": "Tower",
            "first_mes": "Enter.",
            "mes_example": "",
        }
    })
    card = load_character_from_json(json_str)
    assert card.data.name == "Elysia"

def test_load_character_missing_fields_raises():
    with pytest.raises(ValueError):
        load_character_from_dict({"spec": "chara_card_v2"})
```

```python
# tests/story_rp_engine/test_lorebook.py
from story_rp_engine.core.types import Lorebook, LorebookEntry
from story_rp_engine.rp.lorebook import LorebookEngine

def test_lorebook_matching_keywords():
    entry1 = LorebookEntry(keys=["silver sword", "blade"], content="Forged with elven silver.", insertion_order=10)
    entry2 = LorebookEntry(keys=["dragon"], content="Ancient beast of myth.", insertion_order=20)
    lorebook = Lorebook(name="Mythology", entries=[entry1, entry2])

    # Text contains "silver sword"
    matched = LorebookEngine.find_matching_entries([lorebook], "He drew his silver sword quietly.")
    assert len(matched) == 1
    assert matched[0].content == "Forged with elven silver."

def test_lorebook_case_insensitivity():
    entry = LorebookEntry(keys=["Necromancer"], content="Practitioner of dark magic.")
    lorebook = Lorebook(name="Magic", entries=[entry])

    matched = LorebookEngine.find_matching_entries([lorebook], "The necromancer stood tall.")
    assert len(matched) == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_character.py tests/story_rp_engine/test_lorebook.py -v`
Expected: FAIL

- [ ] **Step 3: Implement character parser and lorebook engine**

`src/story_rp_engine/rp/character.py`:
```python
import json
from typing import Any, Dict
from pydantic import ValidationError
from story_rp_engine.core.types import CharacterCardV2

def load_character_from_dict(payload: Dict[str, Any]) -> CharacterCardV2:
    try:
        return CharacterCardV2.model_validate(payload)
    except ValidationError as e:
        raise ValueError(f"Invalid Character Card V2: {e}") from e

def load_character_from_json(json_str: str) -> CharacterCardV2:
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}") from e
    return load_character_from_dict(data)
```

`src/story_rp_engine/rp/lorebook.py`:
```python
import re
from typing import List
from story_rp_engine.core.types import Lorebook, LorebookEntry

class LorebookEngine:
    @staticmethod
    def find_matching_entries(lorebooks: List[Lorebook], text: str) -> List[LorebookEntry]:
        """Finds all lorebook entries whose keys appear as words/phrases in the given text."""
        matched: List[LorebookEntry] = []
        seen_contents = set()

        for book in lorebooks:
            for entry in book.entries:
                if not entry.enabled or entry.content in seen_contents:
                    continue
                for key in entry.keys:
                    pattern = r'\b' + re.escape(key.strip()) + r'\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        matched.append(entry)
                        seen_contents.add(entry.content)
                        break

        # Sort by insertion_order ascending
        matched.sort(key=lambda x: x.insertion_order)
        return matched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/story_rp_engine/test_character.py tests/story_rp_engine/test_lorebook.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/rp/ tests/story_rp_engine/test_character.py tests/story_rp_engine/test_lorebook.py
git commit -m "feat(rp): add character card V2 parser and lorebook engine"
```

---

### Task 4: Roleplay Prompt Assembly Pipeline

**Files:**
- Create: `src/story_rp_engine/rp/prompt_builder.py`
- Test: `tests/story_rp_engine/test_prompt_builder.py`

**Interfaces:**
- Consumes: `CharacterCardV2`, `ChatMessage`, `LorebookEntry`
- Produces:
  - `build_rp_system_instruction(card: CharacterCardV2, active_lore: List[LorebookEntry], user_name: str) -> str`
  - `assemble_history_with_steering(messages: List[ChatMessage], authors_note: Optional[str], depth: int, max_turns: int) -> List[dict]`

- [ ] **Step 1: Write failing test for prompt builder**

```python
# tests/story_rp_engine/test_prompt_builder.py
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data, ChatMessage, LorebookEntry
from story_rp_engine.rp.prompt_builder import (
    build_rp_system_instruction,
    assemble_history_with_steering,
)

def test_build_rp_system_instruction_macro_replacement():
    data = CharacterCardV2Data(
        name="Aria",
        description="{{char}} is an elven scout.",
        personality="Brave",
        scenario="{{user}} meets {{char}} in the woods.",
        first_mes="Halt!",
        mes_example="<START>\n{{user}}: Hello\n{{char}}: Who are you?",
    )
    card = CharacterCardV2(data=data)
    lore = [LorebookEntry(keys=["woods"], content="The Whispering Woods are ancient.")]
    
    instruction = build_rp_system_instruction(card, active_lore=lore, user_name="Alex")
    assert "Aria is an elven scout." in instruction
    assert "Alex meets Aria in the woods." in instruction
    assert "The Whispering Woods are ancient." in instruction
    assert "{{char}}" not in instruction
    assert "{{user}}" not in instruction

def test_assemble_history_with_authors_note_injection():
    messages = [
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Turn 2"),
        ChatMessage(role="user", content="Turn 3"),
        ChatMessage(role="assistant", content="Turn 4"),
    ]
    # Depth = 2 means inserted 2 messages before the end
    assembled = assemble_history_with_steering(
        messages=messages,
        authors_note="[Style: poetic and slow]",
        depth=2,
        max_turns=10
    )
    assert len(assembled) == 5
    # The note should be injected at index len(assembled) - 1 - 2 = 2
    assert assembled[2]["content"] == "[Style: poetic and slow]"
    assert assembled[2]["role"] == "system"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_prompt_builder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement prompt builder**

```python
# src/story_rp_engine/rp/prompt_builder.py
from typing import Dict, List, Optional
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, LorebookEntry

def replace_macros(text: str, char_name: str, user_name: str) -> str:
    if not text:
        return ""
    return text.replace("{{char}}", char_name).replace("{{user}}", user_name)

def build_rp_system_instruction(
    card: CharacterCardV2,
    active_lore: Optional[List[LorebookEntry]] = None,
    user_name: str = "User",
) -> str:
    data = card.data
    char_name = data.name

    sections = [
        f"You are roleplaying as {char_name}. Stay fully in character at all times. Do not break the fourth wall or speak for {user_name}.",
        f"### Character: {char_name}\n"
        f"- Description: {replace_macros(data.description, char_name, user_name)}\n"
        f"- Personality: {replace_macros(data.personality, char_name, user_name)}",
        f"### Scenario\n{replace_macros(data.scenario, char_name, user_name)}",
    ]

    if active_lore:
        lore_snippets = "\n".join([f"- {entry.content}" for entry in active_lore])
        sections.append(f"### World Information & Lore\n{lore_snippets}")

    if data.mes_example:
        sections.append(f"### Dialogue Examples\n{replace_macros(data.mes_example, char_name, user_name)}")

    if data.post_history_instructions:
        sections.append(f"### Additional Directives\n{replace_macros(data.post_history_instructions, char_name, user_name)}")

    return "\n\n".join(sections)

def assemble_history_with_steering(
    messages: List[ChatMessage],
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> List[Dict[str, str]]:
    """Takes recent turns and injects an Author's Note steering directive at a given depth from the end."""
    truncated = messages[-max_turns:]
    history = [{"role": msg.role, "content": msg.content} for msg in truncated]

    if authors_note and authors_note.strip():
        note_dict = {"role": "system", "content": authors_note.strip()}
        if depth <= 0 or depth >= len(history):
            history.append(note_dict)
        else:
            insert_idx = max(0, len(history) - depth)
            history.insert(insert_idx, note_dict)

    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/story_rp_engine/test_prompt_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/rp/prompt_builder.py tests/story_rp_engine/test_prompt_builder.py
git commit -m "feat(rp): add token-aware prompt builder and Author's Note steering"
```

---

### Task 5: Google ADK Roleplay Agent & Session Runner

**Files:**
- Create: `src/story_rp_engine/rp/agent.py`
- Test: `tests/story_rp_engine/test_rp_agent.py`

**Interfaces:**
- Consumes: `CharacterCardV2`, `EngineConfig`, `ChatMessage`, `Lorebook`
- Produces:
  - `create_rp_agent(card: CharacterCardV2, config: EngineConfig, active_lore: List[LorebookEntry], user_name: str) -> LlmAgent`
  - `run_rp_turn(...) -> str`

- [ ] **Step 1: Write failing test for ADK RP agent creation and invocation**

```python
# tests/story_rp_engine/test_rp_agent.py
from unittest.mock import MagicMock, patch
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data, ChatMessage
from story_rp_engine.rp.agent import create_rp_agent, run_rp_turn

def test_create_rp_agent():
    data = CharacterCardV2Data(
        name="Theron", description="Paladin", personality="Noble",
        scenario="Castle gate", first_mes="Stand firm!", mes_example=""
    )
    card = CharacterCardV2(data=data)
    config = EngineConfig(model_name="ollama/llama3.1:8b")

    agent = create_rp_agent(card, config, active_lore=[], user_name="Traveler")
    assert agent.name == "rp-theron"
    assert "Theron" in agent.instruction

@patch("story_rp_agent_mock")
def test_run_rp_turn(mock_call):
    data = CharacterCardV2Data(
        name="Theron", description="Paladin", personality="Noble",
        scenario="Castle gate", first_mes="Stand firm!", mes_example=""
    )
    card = CharacterCardV2(data=data)
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    agent = create_rp_agent(card, config, active_lore=[], user_name="Traveler")

    with patch.object(agent, "invoke", return_value="I guard the realm."):
        reply = run_rp_turn(
            agent=agent,
            history=[],
            user_input="Who are you?",
            authors_note=None
        )
        assert reply == "I guard the realm."
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_rp_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ADK RP agent and turn runner**

```python
# src/story_rp_engine/rp/agent.py
from typing import List, Optional
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, LorebookEntry
from story_rp_engine.rp.prompt_builder import (
    build_rp_system_instruction,
    assemble_history_with_steering,
)

def create_rp_agent(
    card: CharacterCardV2,
    config: EngineConfig,
    active_lore: Optional[List[LorebookEntry]] = None,
    user_name: str = "User",
) -> LlmAgent:
    """Creates a Google ADK LlmAgent configured for character roleplay."""
    model = get_adk_model(config)
    instruction = build_rp_system_instruction(card, active_lore=active_lore, user_name=user_name)
    sanitized_name = f"rp-{card.data.name.lower().replace(' ', '-')}"

    return LlmAgent(
        name=sanitized_name,
        model=model,
        instruction=instruction,
    )

def run_rp_turn(
    agent: LlmAgent,
    history: List[ChatMessage],
    user_input: str,
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> str:
    """Executes a single conversational RP turn through the ADK agent."""
    all_messages = history + [ChatMessage(role="user", content=user_input)]
    assembled = assemble_history_with_steering(all_messages, authors_note=authors_note, depth=depth, max_turns=max_turns)

    # Convert assembled history to prompt turn for the agent
    conversation_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in assembled])
    prompt = f"{conversation_str}\nASSISTANT:"

    response = agent.invoke(prompt)
    if hasattr(response, "text"):
        return str(response.text).strip()
    return str(response).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/story_rp_engine/test_rp_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/rp/agent.py tests/story_rp_engine/test_rp_agent.py
git commit -m "feat(rp): add Google ADK roleplay agent and session runner"
```

---

### Task 6: Story Co-Pilot Multi-Agent Workflow (Director & Writer)

**Files:**
- Create: `src/story_rp_engine/story/__init__.py`
- Create: `src/story_rp_engine/story/director_agent.py`
- Create: `src/story_rp_engine/story/writer_agent.py`
- Create: `src/story_rp_engine/story/workflow.py`
- Test: `tests/story_rp_engine/test_story_workflow.py`

**Interfaces:**
- Consumes: `StoryRequest`, `EngineConfig`
- Produces:
  - `create_director_agent(config: EngineConfig) -> LlmAgent`
  - `create_writer_agent(config: EngineConfig) -> LlmAgent`
  - `expand_story(request: StoryRequest, config: EngineConfig) -> str`

- [ ] **Step 1: Write failing test for Story Director, Writer, and Workflow**

```python
# tests/story_rp_engine/test_story_workflow.py
from unittest.mock import patch
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent
from story_rp_engine.story.workflow import expand_story

def test_create_director_and_writer_agents():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    assert director.name == "story-director"
    assert "framing" in director.instruction.lower()
    assert writer.name == "story-writer"
    assert "prose" in writer.instruction.lower()

def test_expand_story_pipeline():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    request = StoryRequest(
        premise="A detective arrives at a quiet harbor.",
        current_text="Fog covered the docks.",
        instruction="Describe his arrival and first impression.",
        genre="Noir Mystery",
        tone="Dark and brooding",
    )

    with patch("story_rp_engine.story.director_agent.LlmAgent.invoke", return_value="Focus on cold rain and solitary footsteps."):
        with patch("story_rp_engine.story.writer_agent.LlmAgent.invoke", return_value="He stepped into the mist, collar turned against the damp chill."):
            prose = expand_story(request, config)
            assert "mist" in prose
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_story_workflow.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Director, Writer, and Workflow**

`src/story_rp_engine/story/director_agent.py`:
```python
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model

def create_director_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an expert Story Director. Your job is to provide concise scene framing, tonal direction, "
        "and narrative pacing guidance. Given a story premise, existing prose, and user instruction, "
        "output 2-3 brief sentences guiding the Writer on focus, emotional atmosphere, and scene progression."
    )
    return LlmAgent(
        name="story-director",
        model=get_adk_model(config),
        instruction=instruction,
    )
```

`src/story_rp_engine/story/writer_agent.py`:
```python
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model

def create_writer_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an accomplished Fiction Writer. Your task is to write immersive, polished literary prose. "
        "Honor the Director's scene framing, user's instruction, genre, and tone. "
        "Seamlessly continue the existing text without unnecessary preamble or meta-commentary."
    )
    return LlmAgent(
        name="story-writer",
        model=get_adk_model(config),
        instruction=instruction,
    )
```

`src/story_rp_engine/story/workflow.py`:
```python
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent

def expand_story(request: StoryRequest, config: EngineConfig) -> str:
    """Executes the Director -> Writer ADK pipeline to expand story prose."""
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    # Step 1: Director plans scene framing
    director_prompt = (
        f"Premise: {request.premise or 'Not specified'}\n"
        f"Genre: {request.genre}\n"
        f"Tone: {request.tone}\n"
        f"Current Text:\n{request.current_text}\n\n"
        f"User Instruction: {request.instruction}\n"
        "Provide brief scene framing and narrative guidance for the writer."
    )
    director_resp = director.invoke(director_prompt)
    framing = getattr(director_resp, "text", str(director_resp)).strip()

    # Step 2: Writer writes the continuation
    writer_prompt = (
        f"Genre: {request.genre}\n"
        f"Tone: {request.tone}\n"
        f"Director's Guidance: {framing}\n"
        f"User Instruction: {request.instruction}\n\n"
        f"Current Story:\n{request.current_text}\n\n"
        "Write the next prose passage continuing the story:"
    )
    writer_resp = writer.invoke(writer_prompt)
    prose = getattr(writer_resp, "text", str(writer_resp)).strip()

    return prose
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/story_rp_engine/test_story_workflow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/story/ tests/story_rp_engine/test_story_workflow.py
git commit -m "feat(story): add Director and Writer ADK agents and story workflow"
```

---

### Task 7: Storage Layer for Characters, Lorebooks, and Sessions

**Files:**
- Create: `src/story_rp_engine/storage/__init__.py`
- Create: `src/story_rp_engine/storage/store.py`
- Test: `tests/story_rp_engine/test_storage.py`

**Interfaces:**
- Consumes: `CharacterCardV2`, `Lorebook`, `ChatMessage`
- Produces:
  - `EngineStore.save_character(char_id: str, card: CharacterCardV2)`
  - `EngineStore.get_character(char_id: str) -> Optional[CharacterCardV2]`
  - `EngineStore.list_characters() -> Dict[str, CharacterCardV2]`
  - `EngineStore.save_history(session_id: str, messages: List[ChatMessage])`
  - `EngineStore.get_history(session_id: str) -> List[ChatMessage]`

- [ ] **Step 1: Write failing test for storage**

```python
# tests/story_rp_engine/test_storage.py
import pytest
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data, ChatMessage
from story_rp_engine.storage.store import EngineStore

def test_character_crud(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCardV2(data=CharacterCardV2Data(
        name="Valerie", description="Alchemist", personality="Witty",
        scenario="Shop", first_mes="Can I help you?", mes_example=""
    ))
    store.save_character("valerie_1", card)

    retrieved = store.get_character("valerie_1")
    assert retrieved is not None
    assert retrieved.data.name == "Valerie"

    all_chars = store.list_characters()
    assert "valerie_1" in all_chars

def test_session_history(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    history = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
    ]
    store.save_history("session_123", history)

    loaded = store.get_history("session_123")
    assert len(loaded) == 2
    assert loaded[0].content == "Hi"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_storage.py -v`
Expected: FAIL

- [ ] **Step 3: Implement EngineStore**

```python
# src/story_rp_engine/storage/store.py
import json
import os
from typing import Dict, List, Optional
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, Lorebook

class EngineStore:
    def __init__(self, storage_dir: str = ".engine_data"):
        self.storage_dir = storage_dir
        self.char_dir = os.path.join(storage_dir, "characters")
        self.sessions_dir = os.path.join(storage_dir, "sessions")
        os.makedirs(self.char_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)

    def save_character(self, char_id: str, card: CharacterCardV2) -> None:
        path = os.path.join(self.char_dir, f"{char_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(card.model_dump_json(indent=2))

    def get_character(self, char_id: str) -> Optional[CharacterCardV2]:
        path = os.path.join(self.char_dir, f"{char_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CharacterCardV2.model_validate(data)

    def list_characters(self) -> Dict[str, CharacterCardV2]:
        results = {}
        for filename in os.listdir(self.char_dir):
            if filename.endswith(".json"):
                char_id = filename[:-5]
                card = self.get_character(char_id)
                if card:
                    results[char_id] = card
        return results

    def save_history(self, session_id: str, messages: List[ChatMessage]) -> None:
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        data = [msg.model_dump(mode="json") for msg in messages]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_history(self, session_id: str) -> List[ChatMessage]:
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChatMessage.model_validate(item) for item in data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/story_rp_engine/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/storage/ tests/story_rp_engine/test_storage.py
git commit -m "feat(storage): add local file store for characters and session histories"
```

---

### Task 8: FastAPI Service & REST/SSE Streaming Endpoints

**Files:**
- Create: `src/story_rp_engine/api/__init__.py`
- Create: `src/story_rp_engine/api/routes_rp.py`
- Create: `src/story_rp_engine/api/routes_story.py`
- Create: `src/story_rp_engine/api/app.py`
- Test: `tests/story_rp_engine/test_api.py`

**Interfaces:**
- Consumes: All core, RP, Story, and Storage modules.
- Produces: `create_app(store: Optional[EngineStore], config: Optional[EngineConfig]) -> FastAPI`
  - Endpoints:
    - `POST /api/v1/characters`
    - `GET /api/v1/characters`
    - `GET /api/v1/characters/{char_id}`
    - `POST /api/v1/rp/chat`
    - `POST /api/v1/rp/chat/stream`
    - `POST /api/v1/story/expand`
    - `POST /api/v1/story/expand/stream`

- [ ] **Step 1: Write failing test for FastAPI API endpoints**

```python
# tests/story_rp_engine/test_api.py
from fastapi.testclient import TestClient
from unittest.mock import patch
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore

def test_character_endpoints(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    payload = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Lyra", "description": "Minstrel", "personality": "Cheerful",
            "scenario": "Tavern", "first_mes": "Care for a song?", "mes_example": ""
        }
    }
    # Create character
    res = client.post("/api/v1/characters?char_id=lyra", json=payload)
    assert res.status_code == 200

    # Get character
    res = client.get("/api/v1/characters/lyra")
    assert res.status_code == 200
    assert res.json()["data"]["name"] == "Lyra"

    # List characters
    res = client.get("/api/v1/characters")
    assert "lyra" in res.json()

def test_rp_chat_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    # Pre-populate character
    client.post("/api/v1/characters?char_id=lyra", json={
        "spec": "chara_card_v2", "spec_version": "2.0",
        "data": {"name": "Lyra", "description": "A", "personality": "B", "scenario": "C", "first_mes": "D", "mes_example": ""}
    })

    with patch("story_rp_engine.rp.agent.run_rp_turn", return_value="I sing a ballad."):
        res = client.post("/api/v1/rp/chat", json={
            "char_id": "lyra",
            "session_id": "session_1",
            "message": "Play something for us."
        })
        assert res.status_code == 200
        assert res.json()["reply"] == "I sing a ballad."

def test_story_expand_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    with patch("story_rp_engine.story.workflow.expand_story", return_value="The ship docked at dawn."):
        res = client.post("/api/v1/story/expand", json={
            "premise": "A voyage across the sea.",
            "current_text": "The waves were calm.",
            "instruction": "Describe docking.",
        })
        assert res.status_code == 200
        assert res.json()["expansion"] == "The ship docked at dawn."
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/story_rp_engine/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement API routes and application factory**

`src/story_rp_engine/api/routes_rp.py`:
```python
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from story_rp_engine.core.types import CharacterCardV2, ChatMessage
from story_rp_engine.rp.agent import create_rp_agent, run_rp_turn

router = APIRouter(prefix="/api/v1", tags=["Roleplay"])

class RPChatRequest(BaseModel):
    char_id: str
    session_id: str
    message: str
    authors_note: Optional[str] = None
    user_name: Optional[str] = "User"

@router.post("/characters")
def save_character(char_id: str, card: CharacterCardV2, request: Request):
    store = request.app.state.store
    store.save_character(char_id, card)
    return {"status": "saved", "char_id": char_id}

@router.get("/characters")
def list_characters(request: Request):
    store = request.app.state.store
    return store.list_characters()

@router.get("/characters/{char_id}")
def get_character(char_id: str, request: Request):
    store = request.app.state.store
    card = store.get_character(char_id)
    if not card:
        raise HTTPException(status_code=404, detail="Character not found")
    return card

@router.post("/rp/chat")
def chat_rp(req: RPChatRequest, request: Request):
    store = request.app.state.store
    config = request.app.state.config

    card = store.get_character(req.char_id)
    if not card:
        raise HTTPException(status_code=404, detail="Character not found")

    history = store.get_history(req.session_id)
    agent = create_rp_agent(card, config, active_lore=[], user_name=req.user_name or "User")
    reply = run_rp_turn(agent, history, req.message, authors_note=req.authors_note)

    # Persist updated history
    updated_history = history + [
        ChatMessage(role="user", content=req.message),
        ChatMessage(role="assistant", content=reply),
    ]
    store.save_history(req.session_id, updated_history)

    return {"reply": reply, "session_id": req.session_id}

@router.post("/rp/chat/stream")
def chat_rp_stream(req: RPChatRequest, request: Request):
    # Streaming SSE generator wrapper
    res = chat_rp(req, request)
    def event_stream():
        yield f"data: {res['reply']}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

`src/story_rp_engine/api/routes_story.py`:
```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.workflow import expand_story

router = APIRouter(prefix="/api/v1/story", tags=["Story Co-Pilot"])

@router.post("/expand")
def expand_story_endpoint(req: StoryRequest, request: Request):
    config = request.app.state.config
    expansion = expand_story(req, config)
    return {"expansion": expansion}

@router.post("/expand/stream")
def expand_story_stream(req: StoryRequest, request: Request):
    config = request.app.state.config
    expansion = expand_story(req, config)
    def event_stream():
        yield f"data: {expansion}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

`src/story_rp_engine/api/app.py`:
```python
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.api.routes_rp import router as rp_router
from story_rp_engine.api.routes_story import router as story_router

def create_app(
    store: Optional[EngineStore] = None,
    config: Optional[EngineConfig] = None,
) -> FastAPI:
    app = FastAPI(title="Dual-Mode Story & Roleplay Engine API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.store = store or EngineStore()
    app.state.config = config or EngineConfig()

    app.include_router(rp_router)
    app.include_router(story_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/story_rp_engine/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/story_rp_engine/api/ tests/story_rp_engine/test_api.py
git commit -m "feat(api): add FastAPI REST and SSE streaming endpoints"
```

---

### Task 9: End-to-End Test Suite & Verification

**Files:**
- Create: `tests/story_rp_engine/test_e2e_integration.py`

**Interfaces:**
- Exercises the entire stack from Character Card import to RP chat and Story Co-Pilot expansion.

- [ ] **Step 1: Write comprehensive end-to-end integration test**

```python
# tests/story_rp_engine/test_e2e_integration.py
from fastapi.testclient import TestClient
from unittest.mock import patch
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore

def test_full_engine_lifecycle(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    app = create_app(store=store, config=config)
    client = TestClient(app)

    # 1. Health check
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # 2. Register character card
    card_json = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Rowan",
            "description": "A seasoned frontier guide.",
            "personality": "Gruff but reliable.",
            "scenario": "A mountain pass in winter.",
            "first_mes": "Pack your gear tight.",
            "mes_example": "{{user}}: How far?\n{{char}}: Two days, if the snow holds.",
        }
    }
    save_res = client.post("/api/v1/characters?char_id=rowan", json=card_json)
    assert save_res.status_code == 200

    # 3. Conversational RP turn
    with patch("story_rp_engine.rp.agent.run_rp_turn", return_value="Snow is coming. Move faster."):
        chat_res = client.post("/api/v1/rp/chat", json={
            "char_id": "rowan",
            "session_id": "sess_mountain_1",
            "message": "Do you smell snow?"
        })
        assert chat_res.status_code == 200
        assert "Snow is coming" in chat_res.json()["reply"]

    # 4. Verify session persisted
    saved_history = store.get_history("sess_mountain_1")
    assert len(saved_history) == 2
    assert saved_history[0].content == "Do you smell snow?"
    assert saved_history[1].content == "Snow is coming. Move faster."

    # 5. Story Co-Pilot expansion
    with patch("story_rp_engine.story.workflow.expand_story", return_value="The ridge gave way to a vast frozen valley."):
        story_res = client.post("/api/v1/story/expand", json={
            "premise": "Surviving the high winter pass.",
            "current_text": "Rowan tightened the straps on his pack.",
            "instruction": "Describe the view from the pass.",
            "genre": "Adventure",
            "tone": "Gritty",
        })
        assert story_res.status_code == 200
        assert "frozen valley" in story_res.json()["expansion"]
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/story_rp_engine/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/story_rp_engine/test_e2e_integration.py
git commit -m "test: add full lifecycle end-to-end integration test"
```
