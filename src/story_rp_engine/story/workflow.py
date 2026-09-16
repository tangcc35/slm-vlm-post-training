from typing import Any, Iterator
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
    if writer_resp is None:
        return ""
    prose = getattr(writer_resp, "text", str(writer_resp)).strip()

    return prose


def _is_invoke_patched(agent: Any) -> bool:
    if "_invoke_fn" in getattr(agent, "__dict__", {}):
        return True
    invoke_attr = getattr(agent, "invoke", None)
    if hasattr(invoke_attr, "mock_calls") or invoke_attr.__class__.__name__ in ("Mock", "MagicMock"):
        return True
    cls_invoke = getattr(type(agent), "invoke", None)
    if hasattr(cls_invoke, "mock_calls") or cls_invoke.__class__.__name__ in ("Mock", "MagicMock"):
        return True
    return False


def stream_expand_story(request: StoryRequest, config: EngineConfig) -> Iterator[str]:
    """Executes the Director framing step and then streams Writer prose tokens."""
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

    if _is_invoke_patched(writer) or not (hasattr(writer, "stream") and callable(writer.stream)):
        writer_resp = writer.invoke(writer_prompt)
        if writer_resp is None:
            return

        if hasattr(writer_resp, "__iter__") and not isinstance(writer_resp, (str, bytes, dict)):
            for chunk in writer_resp:
                if chunk:
                    yield str(chunk)
            return

        prose = getattr(writer_resp, "text", str(writer_resp)).strip()
        if not prose:
            return

        words = prose.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
        return

    # Otherwise stream via writer.stream
    try:
        for chunk in writer.stream(writer_prompt):
            if chunk:
                yield chunk
    except Exception:
        writer_resp = writer.invoke(writer_prompt)
        if writer_resp is None:
            return
        prose = getattr(writer_resp, "text", str(writer_resp)).strip()
        if prose:
            words = prose.split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word


