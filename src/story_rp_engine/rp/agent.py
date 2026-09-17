from typing import Iterator, List, Optional
from story_rp_engine.core.agent_utils import (
    LlmAgent as _BaseLlmAgent,
    build_adk_agent,
    extract_agent_response_text,
    stream_agent_response,
)
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, LorebookEntry
from story_rp_engine.rp.prompt_builder import (
    assemble_history_with_steering,
    build_rp_system_instruction,
)


class LlmAgent(_BaseLlmAgent):
    """ADK LlmAgent for Roleplay."""
    pass


def create_rp_agent(
    card: CharacterCardV2,
    config: EngineConfig,
    active_lore: Optional[List[LorebookEntry]] = None,
    user_name: str = "User",
) -> LlmAgent:
    """Creates a Google ADK LlmAgent configured for character roleplay."""
    model = get_adk_model(config)
    instruction = build_rp_system_instruction(card, active_lore=active_lore, user_name=user_name)
    sanitized_name = f"rp-{card.data.name.lower().replace(' ', '-')}"

    return build_adk_agent(
        name=sanitized_name,
        model=model,
        instruction=instruction,
        agent_cls=LlmAgent,
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


def run_rp_turn(
    agent: LlmAgent,
    history: List[ChatMessage],
    user_input: str,
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> str:
    """Executes a single conversational RP turn through the ADK agent."""
    prompt = build_rp_turn_prompt(history, user_input, authors_note, depth, max_turns)
    return extract_agent_response_text(agent.invoke(prompt))


def stream_rp_turn(
    agent: LlmAgent,
    history: List[ChatMessage],
    user_input: str,
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> Iterator[str]:
    """Streams partial tokens/chunks for a single conversational RP turn."""
    prompt = build_rp_turn_prompt(history, user_input, authors_note, depth, max_turns)
    return stream_agent_response(agent, prompt)

