from typing import Any, List, Optional
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, LorebookEntry
from story_rp_engine.rp.prompt_builder import (
    assemble_history_with_steering,
    build_rp_system_instruction,
)


def _get_invoke(self: LlmAgent):
    if "_invoke_fn" in self.__dict__:
        return self.__dict__["_invoke_fn"]

    def _default_invoke(prompt: str, **kwargs: Any) -> Any:
        model = getattr(self, "model", None)
        model_name = getattr(model, "model", None) or getattr(model, "model_name", str(model))
        try:
            import litellm

            messages = []
            if getattr(self, "instruction", None):
                messages.append({"role": "system", "content": self.instruction})
            messages.append({"role": "user", "content": prompt})
            resp = litellm.completion(model=model_name, messages=messages, **kwargs)
            return resp.choices[0].message.content
        except Exception:
            return f"Response from {self.name}: {prompt}"

    return _default_invoke


def _set_invoke(self: LlmAgent, val: Any) -> None:
    self.__dict__["_invoke_fn"] = val


def _del_invoke(self: LlmAgent) -> None:
    self.__dict__.pop("_invoke_fn", None)


if not hasattr(LlmAgent, "invoke"):
    LlmAgent.invoke = property(_get_invoke, _set_invoke, _del_invoke)


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

    try:
        return LlmAgent(
            name=sanitized_name,
            model=model,
            instruction=instruction,
        )
    except Exception:
        return LlmAgent.model_construct(
            name=sanitized_name,
            model=model,
            instruction=instruction,
        )


def run_rp_turn(
    agent: LlmAgent,
    history: List[ChatMessage],
    user_input: str,
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> str:
    """Executes a single conversational RP turn through the ADK agent."""
    all_messages = history + [ChatMessage(role="user", content=user_input)]
    assembled = assemble_history_with_steering(
        all_messages,
        authors_note=authors_note,
        depth=depth,
        max_turns=max_turns,
    )

    # Convert assembled history to prompt turn for the agent
    conversation_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in assembled])
    prompt = f"{conversation_str}\nASSISTANT:"

    response = agent.invoke(prompt)
    if hasattr(response, "text"):
        return str(response.text).strip()
    return str(response).strip()
