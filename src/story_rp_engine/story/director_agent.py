from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config


def create_director_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are the Story Director. The conversation holds the user's instructions, your earlier notes, the "
        "passages the Writer produced, and summaries of older turns. Write notes that let the Writer carry out "
        "the user's latest instruction, which may ask to continue, expand, or rewrite part of the story. The "
        "Writer sees only your notes, the story parameters and the recent text below, so include:\n"
        "- What the next passage must accomplish, its beats in order, and where it should stop.\n"
        "- Anything from earlier in the story the Writer needs: names, appearances, places, promises, "
        "unresolved threads.\n\n"
        "Reply with the notes only, kept short.\n\n"
        "### Story Parameters\n"
        "- Premise: {premise?}\n"
        "- Genre: {genre?}\n"
        "- Tone: {tone?}\n\n"
        "### Recent Text\n"
        "{current_text?}"
    )
    return LlmAgent(
        name="story_director",
        model=get_adk_model(config),
        instruction=instruction,
        generate_content_config=get_generate_config(config),
        # Workflow agents default to no history; the director reads the whole (compacted) conversation.
        include_contents="default",
    )
