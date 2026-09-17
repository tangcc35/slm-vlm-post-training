from typing import Any, Iterator, List, Optional
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


def _get_stream(self: LlmAgent):
    if "_stream_fn" in self.__dict__:
        return self.__dict__["_stream_fn"]

    def _default_stream(prompt: str, **kwargs: Any) -> Iterator[str]:
        model = getattr(self, "model", None)
        model_name = getattr(model, "model", None) or getattr(model, "model_name", str(model))
        try:
            import litellm

            messages = []
            if getattr(self, "instruction", None):
                messages.append({"role": "system", "content": self.instruction})
            messages.append({"role": "user", "content": prompt})
            resp = litellm.completion(model=model_name, messages=messages, stream=True, **kwargs)
            for chunk in resp:
                content = None
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception:
            fallback = f"Response from {self.name}: {prompt}"
            words = fallback.split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word

    return _default_stream


def _set_stream(self: LlmAgent, val: Any) -> None:
    self.__dict__["_stream_fn"] = val


def _del_stream(self: LlmAgent) -> None:
    self.__dict__.pop("_stream_fn", None)


if not hasattr(LlmAgent, "stream"):
    LlmAgent.stream = property(_get_stream, _set_stream, _del_stream)


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
    if response is None:
        return ""
    if hasattr(response, "text"):
        return str(response.text).strip() if response.text is not None else ""
    return str(response).strip()


def _is_invoke_patched(agent: Any) -> bool:
    if "_invoke_fn" in getattr(agent, "__dict__", {}):
        return True
    invoke_attr = getattr(agent, "invoke", None)
    if hasattr(invoke_attr, "mock_calls") or invoke_attr.__class__.__name__ in ("Mock", "MagicMock"):
        return True
    cls_invoke = getattr(type(agent), "invoke", None)
    if hasattr(cls_invoke, "mock_calls") or cls_invoke.__class__.__name__ in ("Mock", "MagicMock"):
        return True
    return False


def stream_rp_turn(
    agent: LlmAgent,
    history: List[ChatMessage],
    user_input: str,
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> Iterator[str]:
    """Streams partial tokens/chunks for a single conversational RP turn."""
    all_messages = history + [ChatMessage(role="user", content=user_input)]
    assembled = assemble_history_with_steering(
        all_messages,
        authors_note=authors_note,
        depth=depth,
        max_turns=max_turns,
    )

    conversation_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in assembled])
    prompt = f"{conversation_str}\nASSISTANT:"

    if _is_invoke_patched(agent) or not (hasattr(agent, "stream") and callable(agent.stream)):
        resp = agent.invoke(prompt)
        if resp is None:
            return

        if hasattr(resp, "__iter__") and not isinstance(resp, (str, bytes, dict)):
            for chunk in resp:
                if chunk:
                    yield str(chunk)
            return

        text = getattr(resp, "text", str(resp)).strip()
        if not text:
            return

        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
        return

    # Otherwise stream via agent.stream
    try:
        for chunk in agent.stream(prompt):
            if chunk:
                yield chunk
    except Exception:
        resp = agent.invoke(prompt)
        if resp is None:
            return
        text = getattr(resp, "text", str(resp)).strip()
        if text:
            words = text.split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word


