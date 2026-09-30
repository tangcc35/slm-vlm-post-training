import json
from typing import Any, AsyncIterator, Optional
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.genai import types


def _get_terminal_node_name(agent: Any) -> Optional[str]:
    """Finds the terminal node name in a Workflow graph if present."""
    if hasattr(agent, "graph") and hasattr(agent.graph, "edges") and hasattr(agent.graph, "nodes"):
        from_nodes = {e.from_node.name for e in agent.graph.edges}
        for n in agent.graph.nodes:
            if n.name != "__START__" and n.name not in from_nodes:
                return n.name
    return None


async def execute_runner_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
    state_delta: Optional[dict] = None,
) -> str:
    """Executes a turn asynchronously via ADK Runner and returns final assistant text."""
    content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
    target_node = _get_terminal_node_name(runner.agent)
    final_text = ""
    accumulated_partial = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        state_delta=state_delta,
    ):
        if target_node and event.author and event.author != target_node:
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
) -> AsyncIterator[str]:
    """Streams output token deltas/chunks from a turn asynchronously via ADK Runner."""
    content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
    target_node = _get_terminal_node_name(runner.agent)
    run_cfg = RunConfig(streaming_mode=StreamingMode.SSE)
    yielded_any_partial = False

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        state_delta=state_delta,
        run_config=run_cfg,
    ):
        if target_node and event.author and event.author != target_node:
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


async def format_sse_stream(
    generator: AsyncIterator[str],
    chunk_size: Optional[int] = 4,
) -> AsyncIterator[str]:
    """Buffers string chunks, yielding structured SSE JSON deltas and a final complete text."""
    buffer = []
    full_text_chunks = []
    chunk_threshold = max(1, chunk_size or 4)
    buffered_tokens = 0

    async for chunk in generator:
        if not chunk:
            continue
        buffer.append(chunk)
        full_text_chunks.append(chunk)

        words = chunk.strip().split()
        chunk_tokens = len(words) if words else 1
        buffered_tokens += chunk_tokens

        if chunk_threshold <= 1 or buffered_tokens >= chunk_threshold or len(buffer) >= chunk_threshold or "\n" in chunk:
            combined = "".join(buffer)
            yield f"data: {json.dumps({'delta': combined}, ensure_ascii=False)}\n\n"
            buffer.clear()
            buffered_tokens = 0

    if buffer:
        combined = "".join(buffer)
        yield f"data: {json.dumps({'delta': combined}, ensure_ascii=False)}\n\n"
        buffer.clear()

    full_text = "".join(full_text_chunks)
    yield f"data: {json.dumps({'full_text': full_text, 'done': True}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
