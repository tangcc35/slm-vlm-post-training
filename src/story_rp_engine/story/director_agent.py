from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model


def create_director_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an expert Story Director. Your job is to provide concise scene framing, tonal direction, "
        "and narrative pacing guidance. Given a story premise, existing prose, and user instruction, "
        "output 2-3 brief sentences guiding the Writer on focus, emotional atmosphere, and scene progression."
    )
    return LlmAgent(
        name="story_director",
        model=get_adk_model(config),
        instruction=instruction,
    )



