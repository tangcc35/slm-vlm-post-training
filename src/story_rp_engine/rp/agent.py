import re
from typing import Optional
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config
from story_rp_engine.core.types import CharacterCard
from story_rp_engine.rp.callbacks import rp_before_model_callback
from story_rp_engine.rp.prompt_builder import build_rp_system_instruction


def create_rp_agent(
    card: CharacterCard,
    config: EngineConfig,
    user_name: str = "User",
    **kwargs,
) -> LlmAgent:
    """Creates a Google ADK LlmAgent configured for character roleplay."""
    model = get_adk_model(config)
    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", card.name.lower()).strip("_")

    return LlmAgent(
        name=f"rp_{sanitized_name}",
        model=model,
        # A callable instruction skips ADK's {state_var} templating, which would
        # otherwise raise KeyError on braces in card text such as {{time}}.
        instruction=lambda ctx: build_rp_system_instruction(
            card,
            user_name=ctx.state.get("user_name") or user_name,
            greeting=ctx.state.get("greeting"),
            user_persona=ctx.state.get("user_persona") or "",
        ),
        generate_content_config=get_generate_config(config),
        before_model_callback=rp_before_model_callback,
    )


