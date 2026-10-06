from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config
from story_rp_engine.rp.callbacks import (
    append_to_user_message,
    content_text,
    format_world_info,
    latest_user_index,
    matching_lore,
)


def director_before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Adds the lorebook entries whose keys appear in the user's latest message to that message, so the
    director can pass what the next passage needs on to the writer.

    Like the RP callback, this keeps the system prompt identical across turns.
    """
    user_index = latest_user_index(llm_request)
    if user_index is None:
        return None
    user_text = content_text(llm_request.contents[user_index])
    active_lore = matching_lore(callback_context.state.get("lorebook"), user_text)
    if active_lore:
        note = "[Context for your notes, not shown to the user or the Writer]\n" + format_world_info(active_lore)
        append_to_user_message(llm_request, user_index, note)
    return None


def create_director_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are the Story Director. The conversation holds the user's instructions, your earlier notes, the "
        "passages the Writer produced, and summaries of older turns. Write notes that let the Writer carry out "
        "the user's latest instruction, which may ask to continue, expand, or rewrite part of the story. The "
        "Writer sees only your notes, the story parameters and the recent text below, so include:\n"
        "- What the next passage must accomplish, its beats in order, and where it should stop.\n"
        "- Anything from earlier in the story the Writer needs: names, appearances, places, promises, "
        "unresolved threads.\n"
        "- The world info attached to the user's latest message that the passage needs, restated with its "
        "concrete details. The Writer never sees it otherwise.\n\n"
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
        before_model_callback=director_before_model_callback,
    )
