import re
from typing import List, Optional
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, Lorebook, LorebookEntry
from story_rp_engine.rp.callbacks import create_rp_before_model_callback
from story_rp_engine.rp.prompt_builder import (
    assemble_history_with_steering,
    build_rp_system_instruction,
)


def create_rp_agent(
    card: CharacterCardV2,
    config: EngineConfig,
    lorebook: Optional[Lorebook] = None,
    user_name: str = "User",
    active_lore: Optional[List[LorebookEntry]] = None,
) -> LlmAgent:
    """Creates a Google ADK LlmAgent configured for character roleplay."""
    model = get_adk_model(config)
    instruction = build_rp_system_instruction(card, active_lore=None, user_name=user_name)
    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", card.data.name.lower()).strip("_")
    before_cb = create_rp_before_model_callback(lorebook=lorebook)

    return LlmAgent(
        name=f"rp_{sanitized_name}",
        model=model,
        instruction=instruction,
        before_model_callback=before_cb,
    )


def build_rp_turn_prompt(
    history: List[ChatMessage],
    user_input: str,
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> str:
    """Assembles chat history and steering note into a formatted conversation prompt."""
    all_messages = history + [ChatMessage(role="user", content=user_input)]
    assembled = assemble_history_with_steering(
        all_messages,
        authors_note=authors_note,
        depth=depth,
        max_turns=max_turns,
    )
    conversation_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in assembled])
    return f"{conversation_str}\nASSISTANT:"


