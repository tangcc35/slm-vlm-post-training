from typing import Any
from google.adk.agents import LlmAgent as _BaseLlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model


class LlmAgent(_BaseLlmAgent):
    """Subclass of ADK LlmAgent for Story Director."""
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


def create_director_agent(config: EngineConfig) -> LlmAgent:
    instruction = (
        "You are an expert Story Director. Your job is to provide concise scene framing, tonal direction, "
        "and narrative pacing guidance. Given a story premise, existing prose, and user instruction, "
        "output 2-3 brief sentences guiding the Writer on focus, emotional atmosphere, and scene progression."
    )
    model = get_adk_model(config)
    try:
        return LlmAgent(
            name="story-director",
            model=model,
            instruction=instruction,
        )
    except Exception:
        return LlmAgent.model_construct(
            name="story-director",
            model=model,
            instruction=instruction,
        )
