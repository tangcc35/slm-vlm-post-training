from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent


def format_story_input(request: StoryRequest) -> str:
    """Formats the initial story request into the director node's input prompt."""
    return (
        f"Premise: {request.premise or 'Not specified'}\n"
        f"Genre: {request.genre}\n"
        f"Tone: {request.tone}\n"
        f"Current Text:\n{request.current_text}\n\n"
        f"User Instruction: {request.instruction or 'Continue the story.'}\n"
        "Provide brief scene framing and narrative guidance for the writer."
    )


format_story_director_input = format_story_input


def create_story_workflow(config: EngineConfig) -> Workflow:
    """Creates a declarative ADK Graph Workflow connecting Director and Writer agents."""
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    return Workflow(
        name="story_workflow",
        edges=[
            ("START", director, writer)
        ],
    )

