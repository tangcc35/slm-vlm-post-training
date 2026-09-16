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
