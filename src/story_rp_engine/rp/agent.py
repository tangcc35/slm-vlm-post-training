import re
from typing import Optional
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.core.types import CharacterCardV2
from story_rp_engine.rp.callbacks import rp_before_model_callback
from story_rp_engine.rp.prompt_builder import build_rp_system_instruction


def create_rp_agent(
    card: CharacterCardV2,
    config: EngineConfig,
    user_name: str = "User",
    **kwargs,
) -> LlmAgent:
    """Creates a Google ADK LlmAgent configured for character roleplay."""
    model = get_adk_model(config)
    instruction = build_rp_system_instruction(card, active_lore=None, user_name=user_name)
    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", card.data.name.lower()).strip("_")

    return LlmAgent(
        name=f"rp_{sanitized_name}",
        model=model,
        instruction=instruction,
        before_model_callback=rp_before_model_callback,
    )


