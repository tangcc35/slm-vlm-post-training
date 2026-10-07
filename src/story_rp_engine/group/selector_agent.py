from typing import List, Literal
from google.adk.agents import LlmAgent
from pydantic import create_model
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model, get_generate_config
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.prompt_builder import build_selector_instruction

SPEAKER_SELECTOR = "speaker_selector"


def create_speaker_selector(group: GroupCard, cards: List[CharacterCard], config: EngineConfig) -> LlmAgent:
    """An agent that reads the conversation and returns {"speakers": [char_id, ...]}.

    The schema lists the member IDs, so backends with constrained decoding (Gemini, llama.cpp) can only
    produce members, and any other ID fails validation.
    """
    member_ids = tuple(c.char_id for c in cards)
    speaker_plan = create_model("SpeakerPlan", speakers=(List[Literal[member_ids]], ...))
    return LlmAgent(
        name=SPEAKER_SELECTOR,
        model=get_adk_model(config),
        # Callable, like the RP agent's: card text may contain braces such as {{user}}.
        instruction=lambda ctx: build_selector_instruction(group, cards, user_name=ctx.state.get("user_name") or "User"),
        generate_content_config=get_generate_config(config),
        # Workflow agents default to no history; the selector reads the whole (compacted) conversation.
        include_contents="default",
        output_schema=speaker_plan,
    )
