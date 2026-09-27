# Client-Side Server-Sent Events (SSE) Streaming Reader

Technical reference for client-side Server-Sent Events (SSE) event streaming, ReadableStream decoding, chunk buffering, Markdown rendering, error recovery, and stream interruption in `src/story_rp_engine/web/app.js`.

---

## 1. Overview & Streaming Architecture

The web interface communicates with the FastAPI backend via HTTP `POST` requests returning `text/event-stream` payloads. Unlike native `EventSource` (which is restricted to HTTP `GET`), the frontend employs the Fetch API with `ReadableStream`, `TextDecoder`, and `AbortController`.

```mermaid
sequenceDiagram
    autonumber
    actor User as Browser Client (Vue 3)
    participant API as FastAPI Backend (/stream)
    participant Runner as ADK Agent Runner

    User->>API: POST /api/v1/rp/chat/stream or /api/v1/story/expand/stream
    API-->>User: HTTP 200 (content-type: text/event-stream)
    loop Stream Chunks
        Runner->>API: emit token chunk
        API-->>User: data: {"delta": "..."}\n\n
        User->>User: Buffer & parse event block, append delta, scroll/render
    end
    opt Director Beats Event (Story Mode)
        API-->>User: data: {"director_beats": ["..."]}\n\n
        User->>User: Populate directorBeats accordion
    end
    API-->>User: data: {"full_text": "...", "done": true}\n\n
    API-->>User: data: [DONE]\n\n
    User->>User: reader.read() completes (done: true), finalize state
```

---

## 2. ReadableStream Consumption & Chunk Buffering Pattern

Because TCP packets and network chunks do not align neatly with SSE event frames (`\n\n`), the client must maintain a continuous string buffer to handle fragmented packets and multi-byte UTF-8 sequences.

### Implementation Pattern (`app.js`)

```javascript
const reader = res.body.getReader();
const decoder = new TextDecoder('utf-8');
let buffer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  // Decode incoming Uint8Array with stream: true to preserve multi-byte characters
  buffer += decoder.decode(value, { stream: true });
  buffer = buffer.replace(/\r\n/g, '\n');

  // Split on double newline (SSE event boundary)
  const events = buffer.split('\n\n');
  // Retain incomplete trailing fragment in the buffer
  buffer = events.pop();

  for (const block of events) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    for (const line of trimmed.split('\n')) {
      const l = line.trim();
      if (l.startsWith('data:')) {
        const rawData = l.slice(5).trim();
        if (rawData === '[DONE]') {
          continue; // Termination sentinel
        }

        try {
          const data = JSON.parse(rawData);
          // Handle incremental token deltas
          if (data.delta) {
            targetContainer.text += data.delta;
          } else if (data.full_text && !targetContainer.text) {
            targetContainer.text = data.full_text;
          }
        } catch (jsonErr) {
          // Ignore malformed chunk or partial JSON
        }
      }
    }
  }
}

// Flush any remaining buffered data after stream terminates
if (buffer && buffer.trim()) {
  for (const line of buffer.trim().split('\n')) {
    const l = line.trim();
    if (l.startsWith('data:')) {
      const rawData = l.slice(5).trim();
      if (rawData !== '[DONE]') {
        try {
          const data = JSON.parse(rawData);
          if (data.delta) targetContainer.text += data.delta;
        } catch (_) {}
      }
    }
  }
}
```

---

## 3. Streaming Endpoints Comparison

The application provides two distinct streaming workflows tailored to roleplay conversation turns and narrative story generation:

| Characteristic | Roleplay Chat (`_streamAssistantReply`) | Story Co-Pilot (`expandStory`) |
| :--- | :--- | :--- |
| **Endpoint** | `POST /api/v1/rp/chat/stream` | `POST /api/v1/story/expand/stream` |
| **Request Model** | `RPStreamChatRequest` (`char_id`, `session_id`, `message`, `authors_note`, `user_name`, `chunk_size`) | `StoryStreamExpandRequest` (`session_id`, `premise`, `genre`, `tone`, `current_text`, `instruction`, `chunk_size`) |
| **Target Destination** | Appends to assistant turn in `rpMessages` array | Appends to `storyCurrentText` draft canvas |
| **Preamble Formatting** | Direct character utterance | Leading whitespace insertion (`" "`) if `current_text` does not end with space |
| **Director Metadata** | N/A | Emits `director_beats` array parsed into `this.directorBeats` |
| **Cancellation** | `this.rpAbortController.abort()` | `this.storyAbortController.abort()` |
| **UI Indicator** | Animated `▌` cursor inside active bubble | Animated `▌` cursor below canvas + status text |

---

## 4. Wire Protocol Event Schemas

The client recognizes three standard payload schemas over the SSE channel:

### 1. Token Delta Event
Carries partial token chunks buffered according to the client-requested `chunk_size`:
```text
data: {"delta": "The silver mist drifted silently across the hollow."}

```

### 2. Director Scene Beats Event (Story Mode Only)
Emitted by the multi-agent story graph when the Director Agent finishes outlining beats:
```text
data: {"director_beats": ["Framing: Confrontation at the bridge", "Atmosphere: Tense suspense", "Progression: Guard demands identification"]}

```

### 3. Consolidated Completion Event
Contains the fully assembled generation and status flag:
```text
data: {"full_text": "Complete generated text...", "done": true}

```

### 4. Termination Sentinel
Indicates stream termination; signals client loop completion:
```text
data: [DONE]

```

---

## 5. User Experience & Real-Time Polish

### Smooth Autoscrolling
During chat generation, new tokens expand the feed height. `scrollRPChatToBottom()` runs within Vue's `$nextTick()` to smoothly keep the latest generated text visible:
```javascript
scrollRPChatToBottom() {
  this.$nextTick(() => {
    if (typeof document !== 'undefined') {
      const feed = document.getElementById('rp-chat-feed');
      if (feed) {
        feed.scrollTop = feed.scrollHeight;
      }
    }
  });
}
```

### Real-Time Markdown Rendering (`renderMarkdown`)
Chat bubbles pass raw streamed text through Marked.js:
- Sanitizes and parses markdown headings, lists, bold text, italics, and code fences.
- Provides a secure plain-text fallback replacing HTML entities (`&`, `<`, `>`, `"`) and mapping `\n` to `<br>` if Marked.js is unavailable or parsing fails.

### Seamless Textarea Continuity (Story Co-Pilot)
When continuing an existing story draft, the client avoids awkward glued words by ensuring a boundary space exists between existing text and incoming tokens:
```javascript
if (!appendedInThisTurn && this.storyCurrentText && !/\s$/.test(this.storyCurrentText) && !/^\s/.test(data.delta)) {
  this.storyCurrentText += ' ';
}
this.storyCurrentText += data.delta;
appendedInThisTurn += data.delta;
```

---

## 6. Interruptibility & Session Rewind Mechanics

### Client-Initiated Abort (`AbortController`)
Users can halt token generation immediately by clicking **Stop Generating**:
1. Invokes `controller.abort()`.
2. The fetch stream rejects with `DOMException: The user aborted a request.` (or `AbortError`).
3. The catch block gracefully intercepts `err.name === 'AbortError'`:
   - Preserves whatever partial text was already streamed.
   - Cleans up active generation flags (`isGeneratingRP = false`, `isGeneratingStory = false`).
   - Dismisses pending spinners and restores send button availability.

### Turn Regeneration (`regenerateTurn`)
When regenerating the last assistant response:
1. Locates the preceding user turn in `rpMessages`.
2. Offsets for unpersisted initial greeting (if `rpMessages[0].isGreeting`, `backendIndex = userIdx - 1`).
3. Calls `POST /api/v1/rp/sessions/{id}/turns/delete` with `turn_index: backendIndex, truncate_subsequent: true` to prevent duplicating history on the backend.
4. Removes the assistant bubble from the Vue state.
5. Re-initiates `_streamAssistantReply(priorUserText)`.

### Turn Rewind (`deleteFromHere`)
When clicking "Rewind from here" on turn $N$:
1. Prompts confirmation (`confirm()`).
2. Calls backend turn deletion with `truncate_subsequent: true`.
3. Truncates the reactive frontend message array: `this.rpMessages.splice(index)`.
