from typing import Callable, Optional
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
        lorebooks = [lorebook] if isinstance(lorebook, Lorebook) else list(lorebook)
        active_lore = LorebookEngine.find_matching_entries(lorebooks, user_text)
        if active_lore:
            lore_text = "\n".join([f"- {entry.content}" for entry in active_lore])
            extra_sections.append(f"### Relevant World Information\n{lore_text}")

    # 2. Author's note from session state
    authors_note = None
    if callback_context is not None and getattr(callback_context, "state", None) is not None:
        authors_note = callback_context.state.get("authors_note")
    if authors_note and str(authors_note).strip():
        extra_sections.append(f"### Narrative Directive\n{str(authors_note).strip()}")

    # 3. Inject additions into system instruction
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


def create_rp_before_model_callback(
    lorebook: Optional[Lorebook] = None,
) -> Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]:
    """Deprecated compatibility factory. Use `rp_before_model_callback` directly."""
    if lorebook is None:
        return rp_before_model_callback

    def wrapped_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        if callback_context is not None and getattr(callback_context, "state", None) is not None:
            if "lorebook" not in callback_context.state:
                callback_context.state["lorebook"] = lorebook
        return rp_before_model_callback(callback_context, llm_request)

    return wrapped_callback
