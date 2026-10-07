# User Personas Design

**Date:** 2026-10-07
**Status:** Approved design, pending implementation plan
**Scope:** `src/story_rp_engine/`, `tests/story_rp_engine/`, `tests/test_frontend_integration.py`, `CLAUDE.md`, `.agents/skills/`

## Goal

The user saves personas (who they are in a scene: a name and a description) and picks one when roleplaying with a character or chatting with a group. The characters then know the user's name and who they are.

Success: a user creates a persona in a Personas tab, selects it in the RP or group chat sidebar, and the characters address them by the persona's name and react to its description. Reopening a chat from history reselects its persona.

Out of scope: avatar images, personas in the story tab (it has no "you" in the scene), a default persona or persona locked to a character, persona import/export, and giving the speaker selector the persona description.

## Decisions

- **Personas are saved entities** (`Persona`), managed in their own tab like characters, groups and lorebooks. The chat sidebars only select one.
- **The chat request names the persona; the backend resolves it.** Each turn, `chat_state_delta` loads the persona and copies its name and description into session state, the same way lorebooks are copied. Editing a persona applies to every chat that uses it from that chat's next turn.
- **`user_name` stays** on the chat requests as the fallback when no `persona_id` is sent, so existing API clients (`scripts/story_rp_examples.py`, tests) keep working.
- **The description goes in the system prompt.** It only changes when the user edits or switches personas, so prompt caching still works.

## 1. Backend

### Types (`core/types.py`)

```python
class Persona(BaseModel):
    persona_id: str
    name: str
    description: str = ""
```

`RPChatRequest` and `GroupChatRequest` gain:

```python
persona_id: Optional[str] = Field(default=None, description="Plays as this saved persona; overrides user_name.")
```

### Storage (`storage/store.py`)

`EngineStore` gets a fourth collection next to characters, lorebooks and groups: a `story_rp_personas` table (`persona_id`, `persona_json`, `updated_at`) when a DB URL is set, otherwise JSON files under `<STORY_RP_STORAGE_DIR>/personas/`. It has `save_persona`, `get_persona`, `list_personas` and `delete_persona`, built on the existing `_FileCollection` / `_SqlCollection`.

### Routes (new `api/routes_persona.py`, registered in `api/app.py`)

Copies the group CRUD routes in `api/routes_group.py`; the ID comes from the request body.

- `POST /api/v1/personas`: saves a `Persona`; `400` on an invalid ID.
- `GET /api/v1/personas`: `{persona_id: Persona}`.
- `GET /api/v1/personas/{persona_id}`: `404` if missing.
- `DELETE /api/v1/personas/{persona_id}`: `404` if missing.

### Session state (`api/chat_sessions.py`)

`chat_state_delta` sets these keys on every RP and group turn:

- `persona_id` sent: loads the persona from the store, raising `HTTPException(404, "Persona not found: <id>")` if it's missing (like a missing lorebook), and writes `persona_id`, `user_name = persona.name`, `user_persona = persona.description`.
- `persona_id` not sent: writes `persona_id = None`, `user_name = req.user_name or "User"`, `user_persona = ""`. Switching back to "no persona" clears the description.

`list_chat_sessions` also returns `persona_id` from session state, so the UI can reselect it.

### Prompts

- `rp/prompt_builder.build_rp_system_instruction` and `group/prompt_builder.build_group_system_instruction` take `user_persona: str = ""`. When it is non-empty they add, right after `<scenario>`:

  ```
  <user_persona>
  {description with {{user}} and {{char}} filled}
  </user_persona>
  ```

- `rp/agent.create_rp_agent` and `group/char_agent.create_group_char_agent` pass `user_persona=ctx.state.get("user_persona") or ""` in their instruction callables, next to `user_name`.
- `group/prompt_builder.build_selector_instruction` is unchanged: it keeps only the name.

No `AgentRegistry` change: the instruction callables read session state on every call, so cached agents pick up a new persona without being rebuilt.

### Known limitation

Deleting a persona doesn't touch chats that used it; their session state keeps the copy. When such a chat is reopened, the UI falls back to "no persona", so the next turn sends the user as `"User"`.

## 2. Frontend (`web/index.html`, `web/app.js`)

### Personas tab

A new tab after Lorebooks, laid out like the Groups tab:

- Left column: the saved personas, with New and Delete buttons. No search box.
- Right column: a form with persona ID, name and description. The description's placeholder says `{{user}}` and `{{char}}` macros work. Save posts to `/api/v1/personas`; ID and name are required.

State: `personas` (`{persona_id: Persona}`), `selectedPersonaId`, `personaForm`, plus `loadPersonas`, `selectPersona`, `newPersona`, `savePersona` and `deletePersona` modelled on the group methods. `loadPersonas` runs at startup with the other loaders.

### Chat sidebars

- In both the RP and group chat tabs, the "User Persona Name" text input becomes a **Persona** dropdown: "User (no persona)" (value `''`), then each saved persona by name.
- `rpUserName` / `groupUserName` become `rpPersonaId` / `groupPersonaId` (default `''`). The chat requests send `persona_id` when one is selected and leave it out otherwise; they no longer send `user_name`.
- User message labels show the selected persona's name, or `User`.
- `openRPSession` / `openGroupSession` set the dropdown to the session's `persona_id` if that persona still exists, else `''`.
- New chats keep the current dropdown selection.
- Deleting the persona that's selected in either chat tab resets that dropdown to `''`.

## 3. Tests

- `tests/story_rp_engine/test_storage.py`: persona CRUD, following `test_group_crud`.
- `tests/story_rp_engine/test_prompt_builder.py` and `test_group_prompt_builder.py`: the `<user_persona>` section appears with macros filled, and is absent when the description is empty.
- `tests/story_rp_engine/test_api.py`:
  - persona CRUD routes, including `404` on get/delete of a missing persona;
  - an RP chat with `persona_id` writes `user_name` and `user_persona` to session state and puts the description in the model's system instruction;
  - a later turn without `persona_id` clears `persona_id` and `user_persona`;
  - an unknown `persona_id` returns `404`;
  - `GET /rp/sessions` returns the session's `persona_id`.
- `tests/story_rp_engine/test_group_api.py`: a group chat with `persona_id` puts the persona's name and description in a member's system instruction.
- `tests/test_frontend_integration.py`: the Personas tab is served, following `test_groups_tab_is_served`.

## 4. Docs

- `CLAUDE.md`: add `story_rp_personas` (and personas as JSON files) to the storage paragraph, and `persona_id`, `user_persona` to the RP session state keys.
- `.agents/skills/story-rp-backend/references/api-endpoints-and-sse.md`: document the persona routes and `persona_id` on `RPChatRequest` / `GroupChatRequest`.
- `.agents/skills/story-rp-frontend/references/sse-event-client.md`: add `persona_id` to the request model row.
- `.agents/skills/character-rp/references/character-card-v2-spec.md`: note that `{{user}}` is the selected persona's name when one is set.

`vercel/requirements.txt` is unchanged: no new dependencies.
