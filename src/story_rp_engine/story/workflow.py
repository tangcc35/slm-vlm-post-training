from typing import Iterator
from story_rp_engine.core.agent_utils import (
    extract_agent_response_text,
    stream_agent_response,
)
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent


def build_director_prompt(request: StoryRequest) -> str:
    """Builds the scene framing instruction prompt for the Director."""
    return (
        f"Premise: {request.premise or 'Not specified'}\n"
        f"Genre: {request.genre}\n"
        f"Tone: {request.tone}\n"
        f"Current Text:\n{request.current_text}\n\n"
        f"User Instruction: {request.instruction}\n"
        "Provide brief scene framing and narrative guidance for the writer."
    )


def build_writer_prompt(request: StoryRequest, framing: str) -> str:
    """Builds the continuation prompt for the Writer including Director guidance."""
    return (
        f"Genre: {request.genre}\n"
        f"Tone: {request.tone}\n"
        f"Director's Guidance: {framing}\n"
        f"User Instruction: {request.instruction}\n\n"
        f"Current Story:\n{request.current_text}\n\n"
        "Write the next prose passage continuing the story:"
    )


def expand_story(request: StoryRequest, config: EngineConfig) -> str:
    """Executes the Director -> Writer ADK pipeline to expand story prose."""
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    # Step 1: Director plans scene framing
    director_prompt = build_director_prompt(request)
    framing = extract_agent_response_text(director.invoke(director_prompt))

    # Step 2: Writer writes the continuation
    writer_prompt = build_writer_prompt(request, framing)
    return extract_agent_response_text(writer.invoke(writer_prompt))


def stream_expand_story(request: StoryRequest, config: EngineConfig) -> Iterator[str]:
    """Executes the Director framing step and then streams Writer prose tokens."""
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    # Step 1: Director plans scene framing
    director_prompt = build_director_prompt(request)
    framing = extract_agent_response_text(director.invoke(director_prompt))

    # Step 2: Writer streams the continuation
    writer_prompt = build_writer_prompt(request, framing)
    return stream_agent_response(writer, writer_prompt)

