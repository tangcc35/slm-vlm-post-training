from story_rp_engine.core.agent_utils import LlmAgent as _BaseLlmAgent, build_adk_agent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model


class LlmAgent(_BaseLlmAgent):
    """ADK LlmAgent for Fiction Writer."""
    pass


def create_writer_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an accomplished Fiction Writer. Your task is to write immersive, polished literary prose. "
        "Honor the Director's scene framing, user's instruction, genre, and tone. "
        "Seamlessly continue the existing text without unnecessary preamble or meta-commentary."
    )
    model = get_adk_model(config)
    return build_adk_agent(
        name="story-writer",
        model=model,
        instruction=instruction,
        agent_cls=LlmAgent,
    )

