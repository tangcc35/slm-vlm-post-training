from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config


def create_director_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are the Story Director. Plan the next passage for the Writer, based on the user's instruction "
        "(the message below), the story so far, and the story parameters. The instruction may ask to continue, "
        "expand, or rewrite part of the story.\n\n"
        "Reply with only this short plan:\n"
        "Goal: <what the passage must accomplish>\n"
        "Beats: <2-4 brief beats, in order>\n"
        "POV/tense: <match the existing text>\n"
        "Ending: <where the passage should stop>\n\n"
        "### Story Parameters\n"
        "- Premise: {premise?}\n"
        "- Genre: {genre?}\n"
        "- Tone: {tone?}\n\n"
        "### Existing Text\n"
        "{current_text?}"
    )
    return LlmAgent(
        name="story_director",
        model=get_adk_model(config),
        instruction=instruction,
        generate_content_config=get_generate_config(config),
    )



