import json
import re
from types import SimpleNamespace
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from story_rp_engine.group import char_agent, selector_agent


def _reply(text, partial=False):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]), partial=partial)


@pytest.fixture
def group_models(monkeypatch):
    """Fake models for group chat agents.

    The selector replies with `group_models.plan` (a dict, or a raw string). Each character replies
    "<Name> line", streamed as "<Name> " + "line", and reads its name from its system prompt. Requests are
    recorded in `group_models.requests`, keyed by "speaker_selector" or the character's name.
    """
    fake = SimpleNamespace(plan={"speakers": []}, requests={})

    class SelectorLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            fake.requests.setdefault("speaker_selector", []).append(llm_request)
            yield _reply(fake.plan if isinstance(fake.plan, str) else json.dumps(fake.plan))

    class CharacterLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            name = re.match(r"You are (\S+) in a group roleplay", llm_request.config.system_instruction).group(1)
            fake.requests.setdefault(name, []).append(llm_request)
            if stream:
                yield _reply(f"{name} ", partial=True)
                yield _reply("line", partial=True)
            yield _reply(f"{name} line")

    monkeypatch.setattr(selector_agent, "get_adk_model", lambda config: SelectorLlm())
    monkeypatch.setattr(char_agent, "get_adk_model", lambda config: CharacterLlm())
    return fake
