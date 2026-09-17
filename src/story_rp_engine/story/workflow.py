from typing import Tuple
from google.adk import Workflow
from google.adk.agents import LlmAgent
from story_rp_engine.core.agent_utils import extract_agent_response_text
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


def prepare_story_expansion(
    request: StoryRequest, config: EngineConfig
) -> Tuple[LlmAgent, str]:
    """Executes the Director framing step and returns the Writer agent and continuation prompt."""
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    # Step 1: Director plans scene framing
    director_prompt = build_director_prompt(request)
    framing = extract_agent_response_text(director.invoke(director_prompt))

    # Step 2: Writer continuation prompt
    writer_prompt = build_writer_prompt(request, framing)
    return writer, writer_prompt

