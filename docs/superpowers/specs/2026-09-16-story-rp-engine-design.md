# Dual-Mode Story & Roleplay Engine Design Document

## 1. Overview & Goals

This project designs and implements a **Dual-Mode AI Engine** capable of:
1. **Character Roleplay (RP) Mode:** High-fidelity, 1-on-1 interactive conversation with a user-defined character maintaining strict adherence to persona, tone, style, and memory.
2. **General Story Co-Pilot Mode:** A collaborative creative writing engine that takes user premises, outlines, or prose and expands the narrative with scene framing, stylistic guidance, and seamless continuation.

The system is:
* **Model-Agnostic with Local-First Support:** Built to run on local models (Ollama, vLLM, `llama.cpp`) via OpenAI-compatible APIs and LiteLLM, while retaining full compatibility with cloud models (e.g. Gemini).
* **Orchestrated via Google ADK (`google-adk`):** Leverages Google's Agent Development Kit for multi-agent coordination (Director and Writer) and agent lifecycle management.
* **Modular & Decoupled:** Backend first (Python core engine + FastAPI REST/SSE streaming service), designed for seamless integration with any web frontend.

---

## 2. Architecture & Directory Structure

The engine lives within `src/story_rp_engine/` to avoid interfering with existing SLM/VLM post-training code in the repository.

```text
src/story_rp_engine/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py              # Server settings, local model URLs, sampling parameters
│   ├── model_provider.py      # Factory for ADK LiteLlm instances (Ollama, vLLM, etc.)
│   └── types.py               # Pydantic schemas (CharacterCardV2, Lorebook, Messages, StoryRequest)
├── rp/
│   ├── __init__.py
│   ├── character.py           # Character Card V2 JSON parser and validator
│   ├── lorebook.py            # World Info / keyword matching engine
│   ├── prompt_builder.py      # Token-aware prompt assembler & Author's Note injector
│   └── agent.py               # ADK Character Roleplay Agent & runner
├── story/
│   ├── __init__.py
│   ├── director_agent.py      # ADK Director Agent (narrative framing, tone, scene context)
│   ├── writer_agent.py        # ADK Writer Agent (prose expansion, dialogue, scene generation)
│   └── workflow.py            # ADK Orchestration for Story Co-Pilot (Director -> Writer pipeline)
├── storage/
│   ├── __init__.py
│   └── store.py               # File/SQLite storage for characters, lorebooks, and sessions
└── api/
    ├── __init__.py
    ├── app.py                 # FastAPI application factory with CORS and lifecycle
    ├── routes_rp.py           # Endpoints: /api/v1/characters, /api/v1/rp/chat, /stream
    └── routes_story.py        # Endpoints: /api/v1/story/expand, /api/v1/story/expand/stream
```

---

## 3. Data Models & Schemas

### 3.1 Character Card V2 Specification
Adheres to the standard Community Character Card V2 format:

```python
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
```

### 3.2 Lorebook (World Info)
```python
class LorebookEntry(BaseModel):
    keys: List[str]                  # Trigger keywords (case-insensitive)
    content: str                     # World information snippet
    insertion_order: int = 100       # Priority ordering
    enabled: bool = True

class Lorebook(BaseModel):
    name: str
    description: Optional[str] = None
    entries: List[LorebookEntry] = Field(default_factory=list)
```

### 3.3 Session & Story Models
```python
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
    max_tokens: int = 512
```

---

## 4. Roleplay Engine Architecture

### 4.1 Prompt Assembly Pipeline (`prompt_builder.py`)
To preserve persona fidelity and adherence:
1. **System Directive:** Core anti-repetition, creative writing, and immersion instructions.
2. **Character Definition:** Formats `name`, `personality`, `description`, `scenario`, and `mes_example` (using `{{char}}` and `{{user}}` substitutions).
3. **Lorebook Context Injection:** Scans recent turns (last $N$ messages) for keywords in active Lorebooks, injecting matching entries into context.
4. **Conversation History:** Sliding window maintaining up to $K$ recent turns, respecting maximum context token budgets.
5. **Author's Note / Steering Injection:** Allows steering directives (e.g., "[Style: atmospheric, slow burn]") injected at a configurable depth (e.g. 3 turns before current turn).

### 4.2 ADK Roleplay Agent (`rp/agent.py`)
* Wraps Google ADK's `LlmAgent` using a `LiteLlm` model configuration.
* Manages stateful conversation sessions using ADK's session context.
* Returns responses both synchronously and via asynchronous token event streams.

---

## 5. Story Co-Pilot Architecture (Google ADK Multi-Agent)

### 5.1 Multi-Agent Roles
* **Director Agent (`director_agent.py`):**
  * Evaluates the story premise, genre, tone, and existing prose.
  * Formulates narrative framing, scene context, and tonal direction to guide the story expansion.
* **Writer Agent (`writer_agent.py`):**
  * Focuses strictly on literary prose, pacing, dialogue, and sensory detail.
  * Takes the Director's framing guidance and user instruction to expand the narrative seamlessly.

### 5.2 Workflow Pipeline (`story/workflow.py`)
* **Prose Expansion Workflow (`expand_story`):**
  $$\text{StoryRequest (Premise, Current Text, Instruction, Tone)} \to \text{Director Agent (Scene Framing \& Guidance)} \to \text{Writer Agent} \to \text{Prose Expansion (Streamed or Buffered)}$$

---

## 6. Model Provider & Local Server Integration

### 6.1 `LiteLlm` Factory (`core/model_provider.py`)
* Supports:
  * **Ollama:** `model_name="ollama/<model>"` with `api_base="http://localhost:11434"`.
  * **vLLM / llama.cpp / LocalAI:** `model_name="openai/<model>"` with `api_base="http://localhost:8000/v1"`.
  * **Google Gemini:** `model_name="gemini/<model>"` with `api_key`.
* Centralized sampling parameters: `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `max_tokens`.

---

## 7. API & Streaming Design

### 7.1 Endpoints
* `POST /api/v1/characters` — Create / import Character Card V2.
* `GET /api/v1/characters` — List all stored characters.
* `GET /api/v1/characters/{id}` — Retrieve character details.
* `POST /api/v1/rp/chat` — Generate RP response (buffered).
* `POST /api/v1/rp/chat/stream` — SSE streaming response for interactive chat UI.
* `POST /api/v1/story/expand` — Expand story prose via Director $\to$ Writer workflow (buffered).
* `POST /api/v1/story/expand/stream` — SSE streaming prose expansion for editor canvas.

---

## 8. Error Handling & Edge Cases

* **Local Model Server Unreachable:** Graceful HTTP 503 error with informative diagnostic message ("Cannot connect to model endpoint at http://localhost:11434...").
* **Context Window Overflow:** The prompt preprocessor calculates token counts and automatically truncates older chat history while preserving the Character Card and System Prompts.
* **Corrupt Character Card JSON:** Schema validation using Pydantic raises descriptive 422 Unprocessable Entity errors with field validation details.

---

## 9. Testing Strategy

* **Unit Tests (`tests/story_rp_engine/`):**
  * `test_character_parser.py`: Validates Character Card V2 JSON parsing, field sanitization, and fallback defaults.
  * `test_lorebook.py`: Tests keyword matching, regex triggers, and insertion priority.
  * `test_prompt_builder.py`: Verifies token limits, sliding window truncation, and Author's Note depth placement.
  * `test_model_provider.py`: Tests `LiteLlm` factory and configuration fallback with mocks.
* **Integration Tests:**
  * Mocked ADK agent runner tests verifying complete message roundtrips.
  * FastAPI TestClient verifying REST endpoints and SSE streaming format.
