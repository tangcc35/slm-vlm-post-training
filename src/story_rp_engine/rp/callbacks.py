from typing import List, Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from story_rp_engine.core.types import Lorebook, LorebookEntry
from story_rp_engine.rp.lorebook import LorebookEngine


def latest_user_index(llm_request: LlmRequest) -> Optional[int]:
    for i in range(len(llm_request.contents) - 1, -1, -1):
        content = llm_request.contents[i]
        if content.role == "user" and content.parts:
            return i
    return None


def content_text(content: types.Content) -> str:
    return " ".join(p.text for p in content.parts if getattr(p, "text", None))


def matching_lore(lorebook, text: str) -> List[LorebookEntry]:
    """Entries of the session-state lorebook whose keys appear in text."""
    if not lorebook or not text:
        return []
    # Lorebooks loaded into session state come back from the database as dicts.
    if isinstance(lorebook, (Lorebook, dict)):
        lorebook = [lorebook]
    lorebooks = [Lorebook.model_validate(lb) for lb in lorebook]
    return LorebookEngine.find_matching_entries(lorebooks, text)


def format_world_info(entries: List[LorebookEntry]) -> str:
    return "World info:\n" + "\n".join(f"- {entry.content}" for entry in entries)


def append_to_user_message(llm_request: LlmRequest, index: int, note: str) -> None:
    latest = llm_request.contents[index]
    # Replace rather than mutate: the original Content belongs to the stored session history.
    llm_request.contents[index] = types.Content(role="user", parts=[*latest.parts, types.Part.from_text(text=note)])


def rp_before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """ADK before_model_callback that adds matching lore and the author's note to the latest user message.

    Keeping per-turn context out of the system prompt keeps that prompt identical across turns
    (so prompt caching works) and puts the context next to the message it applies to.
    """
    user_index = latest_user_index(llm_request)
    if user_index is None or callback_context is None:
        return None

    state = callback_context.state
    notes = []

    active_lore = matching_lore(state.get("lorebook"), content_text(llm_request.contents[user_index]))
    if active_lore:
        notes.append(format_world_info(active_lore))

    authors_note = str(state.get("authors_note") or "").strip()
    if authors_note:
        notes.append(f"Author's note: {authors_note}")

    if notes:
        note = "[Context for your next reply, not shown to the user]\n" + "\n\n".join(notes)
        append_to_user_message(llm_request, user_index, note)

    return None
