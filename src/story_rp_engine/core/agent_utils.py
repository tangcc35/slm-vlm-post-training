import json
import logging
from typing import AsyncIterator, Dict, Optional, Tuple, Union
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.genai import types
from story_rp_engine.core.config import EngineConfig

logger = logging.getLogger(__name__)


async def execute_runner_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
    state_delta: Optional[dict] = None,
    author: Optional[str] = None,
) -> str:
    """Executes a turn asynchronously via ADK Runner and returns final assistant text.

    When `author` is set, only that agent's events count (e.g. the story writer, not the director).
    """
    content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
    final_text = ""
    accumulated_partial = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        state_delta=state_delta,
    ):
        if author and event.author != author:
            continue

        text = ""
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
        elif event.output is not None:
            text = str(event.output)

        if text:
            if event.partial:
                accumulated_partial.append(text)
            else:
                final_text = text

    if final_text:
        return final_text.strip()
    if accumulated_partial:
        return "".join(accumulated_partial).strip()
    return ""


async def stream_runner_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
    state_delta: Optional[dict] = None,
    author: Optional[str] = None,
) -> AsyncIterator[str]:
    """Streams output token deltas/chunks from a turn asynchronously via ADK Runner.

    When `author` is set, only that agent's events are streamed.
    """
    content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
    run_cfg = RunConfig(streaming_mode=StreamingMode.SSE)
    yielded_any_partial = False

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        state_delta=state_delta,
        run_config=run_cfg,
    ):
        if author and event.author != author:
            continue

        text = ""
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
        elif event.output is not None:
            text = str(event.output)

        if not text:
            continue

        if event.partial:
            yielded_any_partial = True
            yield text
        else:
            if not yielded_any_partial:
                words = text.split(" ")
                for i, w in enumerate(words):
                    yield w if i == 0 else " " + w


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


async def format_sse_stream(
    generator: AsyncIterator[Union[str, Tuple[str, str]]],
    chunk_size: Optional[int] = 4,
    config: Optional[EngineConfig] = None,
) -> AsyncIterator[str]:
    """Buffers string chunks, yielding structured SSE JSON deltas and a final complete text.

    Group chat streams (speaker, text) pairs instead: each delta then carries its speaker, a new speaker flushes
    the buffer, and the final event lists each speaker's reply in place of full_text. With `config`, a timed-out
    model call is reported by model name.
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
        if isinstance(e, TimeoutError) and config:
            # A model call hit the timeout from get_generate_config; the raw error has no message.
            message = f"{config.model_name} timed out after {config.model_timeout_seconds}s"
        else:
            message = str(e) or type(e).__name__  # the UI ignores an empty error
        yield f"data: {json.dumps({'error': message}, ensure_ascii=False)}\n\n"
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
