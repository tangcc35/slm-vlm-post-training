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
    """ADK before_model_callback that dynamically injects active lore and author's note."""
    user_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                user_text = " ".join([p.text for p in content.parts if getattr(p, "text", None)])
                break

    extra_sections = []

    # 1. Match Lorebook from session state
    lorebook = None
    if callback_context is not None and getattr(callback_context, "state", None) is not None:
        lorebook = callback_context.state.get("lorebook")

    if lorebook and user_text:
        # Lorebooks loaded into session state come back from the database as dicts.
        if isinstance(lorebook, (Lorebook, dict)):
            lorebook = [lorebook]
        lorebooks = [Lorebook.model_validate(lb) for lb in lorebook]
        active_lore = LorebookEngine.find_matching_entries(lorebooks, user_text)
        if active_lore:
            lore_text = "\n".join([f"- {entry.content}" for entry in active_lore])
            extra_sections.append(f"### Relevant World Information\n{lore_text}")

    # 2. Greeting the user saw first; it is not stored as a chat turn
    greeting = callback_context.state.get("greeting") if callback_context is not None else None
    if greeting:
        extra_sections.append(f"### Opening Message (already sent by you)\n{greeting}")

    # 3. Author's note from session state
    authors_note = None
    if callback_context is not None and getattr(callback_context, "state", None) is not None:
        authors_note = callback_context.state.get("authors_note")
    if authors_note and str(authors_note).strip():
        extra_sections.append(f"### Narrative Directive\n{str(authors_note).strip()}")

    # 4. Inject additions into system instruction
    if extra_sections:
        additions = "\n\n".join(extra_sections)
        current_instruction = ""
        if llm_request.config and llm_request.config.system_instruction:
            inst = llm_request.config.system_instruction
            current_instruction = inst if isinstance(inst, str) else str(inst)

        new_instruction = f"{current_instruction}\n\n{additions}".strip()
        if not llm_request.config:
            llm_request.config = types.GenerateContentConfig()
        llm_request.config.system_instruction = new_instruction

    return None
