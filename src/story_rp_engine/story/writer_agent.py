from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config


def create_writer_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an accomplished Fiction Writer. Your task is to write immersive, polished literary prose. "
        "Honor the Director's scene framing, user's instruction, genre ({genre?}), and tone ({tone?}). "
        "Seamlessly continue or modify the story based on prior context and generations without unnecessary preamble or meta-commentary.\n\n"
        "### Story Premise\n"
        "{premise?}\n\n"
        "### User Instruction\n"
        "{instruction?}\n\n"
        "### Existing Text to Continue\n"
        "{current_text?}"
    )
    return LlmAgent(
        name="story_writer",
        model=get_adk_model(config),
        instruction=instruction,
        generate_content_config=get_generate_config(config),
    )



