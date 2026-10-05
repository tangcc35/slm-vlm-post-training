from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from story_rp_engine.core.types import Lorebook
from story_rp_engine.rp.lorebook import LorebookEngine


def rp_before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """ADK before_model_callback that adds matching lore and the author's note to the latest user message.

    Keeping per-turn context out of the system prompt keeps that prompt identical across turns
    (so prompt caching works) and puts the context next to the message it applies to.
    """
    user_index = None
    for i in range(len(llm_request.contents) - 1, -1, -1):
        content = llm_request.contents[i]
        if content.role == "user" and content.parts:
            user_index = i
            break
    if user_index is None or callback_context is None:
        return None

    latest = llm_request.contents[user_index]
    user_text = " ".join(p.text for p in latest.parts if getattr(p, "text", None))
    state = callback_context.state
    notes = []

    lorebook = state.get("lorebook")
    if lorebook and user_text:
        # Lorebooks loaded into session state come back from the database as dicts.
        if isinstance(lorebook, (Lorebook, dict)):
            lorebook = [lorebook]
        lorebooks = [Lorebook.model_validate(lb) for lb in lorebook]
        active_lore = LorebookEngine.find_matching_entries(lorebooks, user_text)
        if active_lore:
            notes.append("World info:\n" + "\n".join(f"- {entry.content}" for entry in active_lore))

    authors_note = str(state.get("authors_note") or "").strip()
    if authors_note:
        notes.append(f"Author's note: {authors_note}")

    if notes:
        note = "[Context for your next reply, not shown to the user]\n" + "\n\n".join(notes)
        # Replace rather than mutate: the original Content belongs to the stored session history.
        llm_request.contents[user_index] = types.Content(
            role="user", parts=[*latest.parts, types.Part.from_text(text=note)]
        )

    return None
