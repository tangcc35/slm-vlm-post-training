from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config


def create_writer_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are a skilled fiction writer working on a {genre?} story with a {tone?} tone. "
        "Write the next passage by following the user's instruction and the Director's notes (the message below). "
        "Unless the instruction asks for something else, such as a rewrite, continue from where the recent text ends.\n\n"
        "- Match the language, point of view, tense and voice of the recent text.\n"
        "- Output only the story text: no preamble, titles or commentary.\n\n"
        "### Story Premise\n"
        "{premise?}\n\n"
        "### User Instruction\n"
        "{instruction?}\n\n"
        "### Recent Text (the end of the story so far)\n"
        "{current_text?}"
    )
    return LlmAgent(
        name="story_writer",
        model=get_adk_model(config),
        instruction=instruction,
        generate_content_config=get_generate_config(config),
    )
