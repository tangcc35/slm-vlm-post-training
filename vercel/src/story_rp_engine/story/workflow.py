from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent


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

