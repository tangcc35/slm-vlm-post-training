from typing import List
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.prompt_builder import build_group_system_instruction, group_agent_name
from story_rp_engine.rp.callbacks import rp_before_model_callback


def create_group_char_agent(
    card: CharacterCard,
    group: GroupCard,
    cards: List[CharacterCard],
    config: EngineConfig,
) -> LlmAgent:
    """A group member's agent: it reads the whole conversation and writes only its own character's reply."""
    return LlmAgent(
        name=group_agent_name(card.char_id),
        model=get_adk_model(config),
        # Callable, like the RP agent's: card text may contain braces such as {{user}}.
        instruction=lambda ctx: build_group_system_instruction(
            card,
            group,
            cards,
            user_name=ctx.state.get("user_name") or "User",
            user_persona=ctx.state.get("user_persona") or "",
        ),
        generate_content_config=get_generate_config(config),
        include_contents="default",
        before_model_callback=rp_before_model_callback,
    )
