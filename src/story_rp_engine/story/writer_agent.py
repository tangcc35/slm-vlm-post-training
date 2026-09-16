from typing import Any
from google.adk.agents import LlmAgent as _BaseLlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model


class LlmAgent(_BaseLlmAgent):
    """Subclass of ADK LlmAgent for Story Writer."""
    pass


def _get_invoke(self: Any):
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


def _set_invoke(self: Any, val: Any) -> None:
    self.__dict__["_invoke_fn"] = val


def _del_invoke(self: Any) -> None:
    self.__dict__.pop("_invoke_fn", None)


if not hasattr(_BaseLlmAgent, "invoke"):
    _BaseLlmAgent.invoke = property(_get_invoke, _set_invoke, _del_invoke)


def _get_stream(self: Any):
    if "_stream_fn" in self.__dict__:
        return self.__dict__["_stream_fn"]

    def _default_stream(prompt: str, **kwargs: Any):
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


def _set_stream(self: Any, val: Any) -> None:
    self.__dict__["_stream_fn"] = val


def _del_stream(self: Any) -> None:
    self.__dict__.pop("_stream_fn", None)


if not hasattr(_BaseLlmAgent, "stream"):
    _BaseLlmAgent.stream = property(_get_stream, _set_stream, _del_stream)


def create_writer_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an accomplished Fiction Writer. Your task is to write immersive, polished literary prose. "
        "Honor the Director's scene framing, user's instruction, genre, and tone. "
        "Seamlessly continue the existing text without unnecessary preamble or meta-commentary."
    )
    model = get_adk_model(config)
    try:
        return LlmAgent(
            name="story-writer",
            model=model,
            instruction=instruction,
        )
    except Exception:
        return LlmAgent.model_construct(
            name="story-writer",
            model=model,
            instruction=instruction,
        )
