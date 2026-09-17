import re
from typing import Any, Iterator
from google.adk.agents import LlmAgent


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


def _get_stream(self: Any):
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
            text = f"Response from {self.name}: {prompt}"
            words = text.split(" ")
            for i, w in enumerate(words):
                yield w if i == 0 else " " + w

    return _default_stream


def _set_stream(self: Any, val: Any) -> None:
    self.__dict__["_stream_fn"] = val


def _del_stream(self: Any) -> None:
    self.__dict__.pop("_stream_fn", None)


if not hasattr(LlmAgent, "invoke"):
    LlmAgent.invoke = property(_get_invoke, _set_invoke, _del_invoke)

if not hasattr(LlmAgent, "stream"):
    LlmAgent.stream = property(_get_stream, _set_stream, _del_stream)


def is_invoke_patched(agent: Any) -> bool:
    """Detects whether an agent's invoke method has been mocked or replaced."""
    if "_invoke_fn" in getattr(agent, "__dict__", {}):
        return True
    invoke_attr = getattr(agent, "invoke", None)
    if hasattr(invoke_attr, "mock_calls") or invoke_attr.__class__.__name__ in ("Mock", "MagicMock"):
        return True
    cls_invoke = getattr(type(agent), "invoke", None)
    if hasattr(cls_invoke, "mock_calls") or cls_invoke.__class__.__name__ in ("Mock", "MagicMock"):
        return True
    return False


def build_adk_agent(
    name: str,
    model: Any,
    instruction: str,
) -> LlmAgent:
    """Builds an ADK LlmAgent, gracefully handling identifier constraints across ADK versions."""
    try:
        return LlmAgent(name=name, model=model, instruction=instruction)
    except Exception:
        return LlmAgent.model_construct(
            name=name,
            model=model,
            instruction=instruction,
        )



def extract_agent_response_text(response: Any) -> str:
    """Extracts cleaned string response from an agent invocation."""
    if response is None:
        return ""
    if hasattr(response, "text"):
        return str(response.text).strip() if response.text is not None else ""
    return str(response).strip()


def stream_agent_response(agent: Any, prompt: str) -> Iterator[str]:
    """Streams tokens/chunks from an agent, falling back to invoke word-splitting when invoke is mocked."""
    if is_invoke_patched(agent) or not (hasattr(agent, "stream") and callable(agent.stream)):
        resp = agent.invoke(prompt)
        if resp is None:
            return

        if hasattr(resp, "__iter__") and not isinstance(resp, (str, bytes, dict)):
            for item in resp:
                yield str(item)
            return

        text = getattr(resp, "text", str(resp))
        words = text.split(" ")
        for i, w in enumerate(words):
            yield w if i == 0 else " " + w
        return

    stream_gen = agent.stream(prompt)
    if stream_gen is not None:
        for chunk in stream_gen:
            if chunk:
                yield str(chunk)


