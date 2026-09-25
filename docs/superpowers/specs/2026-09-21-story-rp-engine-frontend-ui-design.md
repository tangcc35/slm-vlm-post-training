# Story RP Engine Frontend UI Design Specification

**Date:** 2026-09-21  
**Status:** Approved  
**Approach:** Linear Two-Column Workbench (Playground Style, Zero-Build SPA)

---

## 1. Overview & Goals

The `story_rp_engine` is a dual-mode system combining:
1. **Roleplay Mode (RP)**: Character-driven conversational agents with personality, scenario, first message, alternate greetings, author's note steering, and keyword-activated lorebooks.
2. **Story Mode (Co-Pilot)**: Multi-turn creative writing co-pilot powered by an ADK workflow (Director Agent outlining scene beats $\rightarrow$ Writer Agent drafting prose).

The objective is to provide a clean, responsive, user-friendly frontend UI that:
- Runs with **zero build steps** (no Node.js/npm required; served directly by FastAPI via CDN-loaded modern libraries).
- Follows **Approach C (Linear Two-Column Workbench)**: an intuitive, playground-style UI with parameters on the left and outputs/feed on the right.
- Makes **every single element** of the engine's data models ([`CharacterCard`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L5-L18), [`Lorebook`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L27-L31), [`StoryRequest`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L39-L48), and [`RPChatRequest`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L50-L57)) fully editable on the UI without unnecessary UI bloat.
- Supports **real-time SSE streaming** for both RP chat and story prose expansion.
- Supports **chat turn deletion** and conversation rewinding to prune unwanted branches.

---

## 2. Technology Stack & Delivery Model

### Zero-Build Modern SPA
- **Host / Serving**: FastAPI `StaticFiles` mounted at `/` in [`src/story_rp_engine/api/app.py`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/api/app.py). Running `./run_sh/run_story_rp_backend.sh` immediately serves the UI at `http://localhost:8000/`.
- **UI Framework**: Vue 3 (ESM / global build via unpkg CDN: `https://unpkg.com/vue@3/dist/vue.global.prod.js`).
- **Styling**: Tailwind CSS (CDN: `https://cdn.tailwindcss.com`) with a dark-mode palette, clean typography, and responsive grid layouts.
- **Icons**: Lucide Icons (CDN: `https://unpkg.com/lucide@latest`).
- **Markdown Rendering**: Marked.js (CDN: `https://cdn.jsdelivr.net/npm/marked/marked.min.js`) for rich rendering of character dialogue and story prose.

### File Structure
```
src/story_rp_engine/
├── api/
│   ├── app.py                  # Mounts static UI directory at /
│   ├── routes_rp.py            # RP chat, characters CRUD, and turn deletion
│   ├── routes_story.py         # Story expansion streaming
│   └── routes_lorebook.py      # Lorebook CRUD endpoints
└── web/
    ├── index.html              # HTML5 single-page application shell
    ├── app.js                  # Reactive Vue 3 application logic
    └── style.css               # Minimal typography & scrollbar polish
```

---

## 3. Backend API Enhancements

To ensure every element of the engine can be manipulated from the frontend, the following API endpoints will be added or updated:

### 3.1 Lorebook Management (`/api/v1/lorebooks`)
- `GET /api/v1/lorebooks`: Returns dictionary/list of all saved lorebooks.
- `GET /api/v1/lorebooks/{lorebook_id}`: Returns the complete [`Lorebook`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L27-L31) schema.
- `POST /api/v1/lorebooks`: Creates or updates a Lorebook (persisted to `.engine_data/lorebooks/{id}.json`).
- `DELETE /api/v1/lorebooks/{lorebook_id}`: Deletes a Lorebook file.

### 3.2 Character Card Deletion (`DELETE /api/v1/characters/{char_id}`)
- `DELETE /api/v1/characters/{char_id}`: Deletes the character card JSON file from `.engine_data/characters/` and purges any cached runner in `AgentRegistry`.

### 3.3 Roleplay Session History & Turn Deletion (`/api/v1/rp/sessions`)
- `GET /api/v1/rp/sessions/{session_id}/turns`: Retrieves past messages/events recorded in `DatabaseSessionService` for that session.
- `POST /api/v1/rp/sessions/{session_id}/turns/delete`: Deletes a specific turn or truncates history back to a given turn index, synchronizing ADK memory.
- `DELETE /api/v1/rp/sessions/{session_id}`: Clears the session entirely.

---

## 4. Workbench View Specifications

The UI provides a unified top navigation bar with 4 tabs:
`[ 💬 Roleplay ]` | `[ ✍️ Story Co-Pilot ]` | `[ 👤 Characters ]` | `[ 📖 Lorebooks ]`
along with a live backend health indicator badge.

---

### Tab 1: Roleplay Workbench (`/` -> Roleplay Tab)

A linear two-column layout:

#### Left Column: Parameters & Context
- **Character Picker**: Dropdown populated dynamically via `GET /api/v1/characters`.
- **Alternate Greetings Switcher**: If the selected character card has `alternate_greetings`, a dropdown allows selecting the opener (or default `first_mes`).
- **Session ID**: Editable text field with:
  - `New Session` button (generates a random UUID).
  - `Clear History` button (resets current chat).
- **User Name / Persona**: Input field (default: `"User"`) sent as `user_name` in [`RPChatRequest`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L50-L57).
- **Author's Note**: Textarea to inject high-priority guidance (e.g. `[Pacing: slow, romantic tension; Focus on sensory details]`), sent with each turn as `authors_note`.
- **Active Lorebook**: Dropdown to associate a saved Lorebook with the RP session.
- **Chunk Size Slider**: Range from 1 to 64 (default 16) controlling token buffering before SSE emissions.

#### Right Column: Live Chat & Dialogue Feed
- **Dialogue Feed**:
  - Distinct message bubbles: Assistant (left) with avatar placeholder/name, User (right).
  - Markdown formatting rendered via Marked.js.
  - Initial greeting displayed automatically when a character is selected.
- **Turn Actions (per bubble)**:
  - **Delete Turn (Trash Icon)**: Removes the specific message turn and synchronizes with the backend.
  - **Delete From Here**: Truncates all subsequent turns to branch or rewind the dialogue.
  - **Regenerate**: Available on assistant turns to re-roll the reply.
  - **Copy**: Copies message text to clipboard.
- **Input Area**:
  - Multiline auto-expanding textarea (`Enter` to send, `Shift+Enter` for newline).
  - `Send` button (disabled while generating or input is empty).
  - `Stop Generating` button (aborts active fetch `AbortController`).

---

### Tab 2: Story Co-Pilot Workbench (`/` -> Story Tab)

A linear two-column layout:

#### Left Column: Story Controls & Directives
- **Session ID**: Text input with `New Story` button to manage multi-turn story state.
- **Premise**: Textarea for overarching story premise / world premise.
- **Genre**: Select dropdown with presets (*Fiction, Fantasy, Sci-Fi, Cyberpunk, Mystery, Horror, Romance, Slice of Life*) + text input for custom genres.
- **Tone**: Select dropdown with presets (*Balanced, Gritty, Dark, Whimsical, Tense, Poetic, Sensual, Epic*) + text input for custom tones.
- **Next Turn Instruction**: Textarea instructing the engine what to write next (default: `"Continue the story naturally from the current point."`).
- **Max Tokens Slider**: Range 64 to 2048 (default 512).
- **Chunk Size Slider**: Range 1 to 64 (default 16).
- **Triggers**:
  - **Expand Story** button (primary streaming action).
  - **Stop Generating** button (cancels streaming via `AbortController`).

#### Right Column: Story Document & Streaming Canvas
- **Story Document Area**:
  - Large, distraction-free text editor displaying `current_text`.
  - User can freely type, edit, rephrase, or paste text at any time.
  - When "Expand Story" is clicked, SSE streaming appends tokens directly to the text in real-time with an active cursor indicator (`▌`).
- **Document Toolbar**:
  - Word counter and estimated token count.
  - **Undo Last Expansion**: Restores `current_text` to the state prior to the last generation.
  - **Copy Draft**: Copies entire document to clipboard.
  - **Export .md / .txt**: Downloads the draft as a local text file.
- **Collapsible Director Beats Inspector**:
  - Accordion panel below the text area displaying the planned scene beats from the ADK Director Agent.

---

### Tab 3: Characters Management Workbench (`/` -> Characters Tab)

A linear two-column layout:

#### Left Column: Character Directory
- Search input to filter existing characters by name or tag.
- List of saved character cards with character name and tag chips.
- Action buttons:
  - `+ New Character`: Resets the form on the right to empty defaults.
  - `Import JSON`: File picker supporting SillyTavern / Character Card V2 JSON files.
  - `Export JSON`: Downloads the active character as a JSON file.

#### Right Column: Character Form Editor
Cleanly exposes every field from [`CharacterCard`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L5-L18):
- **Identity**:
  - `char_id`: Unique alphanumeric ID / filename slug (required).
  - `name`: Character display name (required).
  - `tags`: Dynamic tag chips input (add/remove tags).
- **Persona & World**:
  - `description`: Character visual summary, background, and biography.
  - `personality`: Core personality traits, behavioral nuances, and speech style.
  - `scenario`: Setting, relationship context, and current environment.
- **Dialogue & Openers**:
  - `first_mes`: Default opening greeting.
  - `alternate_greetings`: Dynamic list of additional openers with `+ Add Greeting`, edit, and delete per greeting.
  - `mes_example`: Few-shot dialogue examples (`<START>\n{{user}}: ...\n{{char}}: ...`).
- **Advanced System Prompts**:
  - `system_prompt`: Optional system instruction override.
  - `post_history_instructions`: Optional instructions appended after conversation history.
  - `creator_notes`: Author notes and instructions.
- **Form Actions**:
  - `Save Character` button (calls `POST /api/v1/characters`).
  - `Delete Character` button with modal confirmation (calls `DELETE /api/v1/characters/{char_id}`).

---

### Tab 4: Lorebooks Management Workbench (`/` -> Lorebooks Tab)

A linear two-column layout:

#### Left Column: Lorebook Directory
- Search filter for saved lorebooks.
- List of saved lorebooks with entry counts.
- Action buttons:
  - `+ New Lorebook`: Resets form to create a new lorebook.
  - `Import JSON`: Imports existing Lorebook JSON file.
  - `Export JSON`: Downloads the active lorebook as JSON.

#### Right Column: Lorebook & Entries Editor
Cleanly exposes [`Lorebook`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L27-L31) and [`LorebookEntry`](file:///home/tangc/slm-vlm-post-training/src/story_rp_engine/core/types.py#L20-L25):
- **Lorebook Header**:
  - `name`: Title of the lorebook (required).
  - `description`: Overview description of the setting/world.
- **Entries List**:
  - Dynamic list of entries with cards displaying:
    - **Active Toggle (`enabled`)**: Checkbox to toggle inclusion.
    - **Insertion Order (`insertion_order`)**: Number input (default 100).
    - **Trigger Keys (`keys`)**: Comma-separated or tag chips input of keyword triggers.
    - **Content (`content`)**: Multiline textarea with world lore / details.
    - **Delete Entry**: Removes the specific entry.
  - `+ Add New Entry` button to append a fresh entry to the list.
- **Form Actions**:
  - `Save Lorebook` button (calls `POST /api/v1/lorebooks`).
  - `Delete Lorebook` button (calls `DELETE /api/v1/lorebooks/{id}`).

---

## 5. Error Handling & Edge Cases

1. **Backend Connectivity**:
   - The UI polls `GET /health` periodically (every 15s). A subtle status pill in the header displays `Backend: Online` (green) or `Backend: Disconnected` (red).
   - If a request fails, an inline toast or banner shows the exact backend error message without breaking the current form or chat state.
2. **Streaming Disconnection & Abort**:
   - Both RP and Story streaming use `AbortController`. Clicking `Stop Generating` immediately cancels the fetch request and closes the SSE stream gracefully.
   - If the connection drops mid-stream, accumulated text is preserved, and an error notice is appended.
3. **Turn Deletion Consistency**:
   - When a turn is deleted on the UI, the frontend issues a delete request to `/api/v1/rp/sessions/{session_id}/turns/delete` with the turn index so subsequent prompts from the ADK Runner do not retain discarded turns.
4. **Validation**:
   - Form inputs enforce required fields (`char_id`, `name`, `session_id`) with clear inline alerts before submission.

---

## 6. Verification & Testing Strategy

1. **API Endpoint Unit / Integration Tests**:
   - Test `GET /`, verifying that `index.html` is returned with `200 OK`.
   - Test Lorebook CRUD endpoints (`POST /api/v1/lorebooks`, `GET /api/v1/lorebooks`, `DELETE /api/v1/lorebooks/{id}`).
   - Test Character deletion endpoint (`DELETE /api/v1/characters/{char_id}`).
   - Test session turn deletion endpoint (`POST /api/v1/rp/sessions/{session_id}/turns/delete`).
2. **Frontend UI End-to-End Verification**:
   - Verify all 4 tabs load correctly and switch cleanly.
   - Verify character creation, editing, and saving.
   - Verify lorebook entry creation, key editing, and saving.
   - Verify Roleplay streaming chat, bubble rendering, and turn deletion.
   - Verify Story Co-Pilot text input, streaming expansion, and undo functionality.
