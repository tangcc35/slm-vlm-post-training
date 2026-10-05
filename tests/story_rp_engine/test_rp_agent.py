from types import SimpleNamespace
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, Lorebook, LorebookEntry
from story_rp_engine.rp.agent import create_rp_agent


def test_create_rp_agent():
    card = CharacterCard(
        char_id="theron",
        name="Theron",
        description="Paladin",
        personality="Noble",
        scenario="Castle gate",
        first_mes="Stand firm!",
        mes_example="",
    )
    config = EngineConfig(model_name="ollama/llama3.1:8b")

    agent = create_rp_agent(card, config, user_name="Traveler")
    assert agent.name == "rp_theron"
    assert "Theron" in agent.instruction(SimpleNamespace(state={}))
    assert "Paladin" in agent.instruction(SimpleNamespace(state={}))
    assert "Traveler" in agent.instruction(SimpleNamespace(state={}))
    assert agent.before_model_callback is not None


def test_create_rp_agent_with_lore():
    card = CharacterCard(
        char_id="theron",
        name="Theron",
        description="Paladin",
        personality="Noble",
        scenario="Castle gate",
        first_mes="Stand firm!",
        mes_example="",
    )
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    lorebook = Lorebook(
        name="chivalry",
        entries=[
            LorebookEntry(keys=["silver"], content="Silver Order of Knights"),
        ],
    )

    agent = create_rp_agent(card, config, lorebook=lorebook, user_name="Traveler")
    assert agent.name == "rp_theron"
    assert agent.before_model_callback is not None
    assert "Silver Order of Knights" not in agent.instruction(SimpleNamespace(state={}))




async def _run_rp_agent(card, config):
    """Runs one RP turn with a recording model; returns the request the model received."""
    seen = []

    class RecordingLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            seen.append(llm_request)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text="Hi.")]))

    agent = create_rp_agent(card, config)
    agent.model = RecordingLlm()
    runner = Runner(app_name="rp_app", agent=agent, session_service=InMemorySessionService(), auto_create_session=True)
    async for _ in runner.run_async(
        user_id="User",
        session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="hello")]),
    ):
        pass
    return seen[0]


@pytest.mark.anyio
async def test_rp_agent_passes_card_braces_through_literally():
    card = CharacterCard(char_id="ava", name="Ava", description="Checks {{time}} often. Hates {location}.")
    request = await _run_rp_agent(card, EngineConfig(model_name="ollama/llama3.1:8b"))
    assert "Checks {{time}} often. Hates {location}." in request.config.system_instruction


@pytest.mark.anyio
async def test_rp_agent_uses_configured_sampling():
    config = EngineConfig(temperature=0.3, top_p=0.5, max_tokens=700)
    cfg = (await _run_rp_agent(CharacterCard(char_id="ava", name="Ava"), config)).config
    assert (cfg.temperature, cfg.top_p, cfg.max_output_tokens) == (0.3, 0.5, 700)


@pytest.mark.anyio
async def test_rp_agent_sends_no_output_cap_by_default(monkeypatch):
    monkeypatch.delenv("STORY_RP_MAX_TOKENS", raising=False)
    request = await _run_rp_agent(CharacterCard(char_id="ava", name="Ava"), EngineConfig())
    assert request.config.max_output_tokens is None
