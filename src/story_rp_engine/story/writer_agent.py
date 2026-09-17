from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model


def create_writer_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an accomplished Fiction Writer. Your task is to write immersive, polished literary prose. "
        "Honor the Director's scene framing, user's instruction, genre, and tone. "
        "Seamlessly continue the existing text without unnecessary preamble or meta-commentary."
    )
    return LlmAgent(
        name="story_writer",
        model=get_adk_model(config),
        instruction=instruction,
    )



