from typing import Callable, Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from story_rp_engine.core.types import Lorebook
from story_rp_engine.rp.lorebook import LorebookEngine


def create_rp_before_model_callback(
    lorebook: Optional[Lorebook] = None,
) -> Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]:
    """Creates an ADK before_model_callback that injects active lore and author's note."""

    def rp_before_model_callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        # Extract user input from latest user content
        user_text = ""
        if llm_request.contents:
            for content in reversed(llm_request.contents):
                if content.role == "user" and content.parts:
                    user_text = " ".join([p.text for p in content.parts if getattr(p, "text", None)])
                    break

        extra_sections = []

        # 1. Match Lorebook entries
        if lorebook and user_text:
            active_lore = LorebookEngine.find_matching_entries([lorebook], user_text)
            if active_lore:
                lore_text = "\n".join([f"- {entry.content}" for entry in active_lore])
                extra_sections.append(f"### Relevant World Information\n{lore_text}")

        # 2. Author's note from session state
        authors_note = None
        if callback_context and callback_context.state:
            authors_note = callback_context.state.get("authors_note")
        if authors_note and str(authors_note).strip():
            extra_sections.append(f"### Narrative Directive\n{str(authors_note).strip()}")

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

    return rp_before_model_callback
