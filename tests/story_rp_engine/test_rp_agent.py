from unittest.mock import MagicMock
from story_rp_engine.core.agent_utils import extract_agent_response_text, stream_agent_response
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data, ChatMessage, Lorebook, LorebookEntry
from story_rp_engine.rp.agent import build_rp_turn_prompt, create_rp_agent


def test_create_rp_agent():
    data = CharacterCardV2Data(
        name="Theron",
        description="Paladin",
        personality="Noble",
        scenario="Castle gate",
        first_mes="Stand firm!",
        mes_example="",
    )
    card = CharacterCardV2(data=data)
    config = EngineConfig(model_name="ollama/llama3.1:8b")

    agent = create_rp_agent(card, config, user_name="Traveler")
    assert agent.name == "rp_theron"
    assert "Theron" in agent.instruction
    assert "Paladin" in agent.instruction
    assert "Traveler" in agent.instruction
    assert agent.before_model_callback is not None


def test_create_rp_agent_with_lore():
    data = CharacterCardV2Data(
        name="Theron",
        description="Paladin",
        personality="Noble",
        scenario="Castle gate",
        first_mes="Stand firm!",
        mes_example="",
    )
    card = CharacterCardV2(data=data)
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
    assert "Silver Order of Knights" not in agent.instruction


def test_build_rp_turn_prompt():
    prompt = build_rp_turn_prompt(
        history=[],
        user_input="Who are you?",
        authors_note=None,
    )
    assert "USER: Who are you?" in prompt
    assert prompt.endswith("\nASSISTANT:")


def test_build_rp_turn_prompt_with_history_and_authors_note():
    history = [
        ChatMessage(role="user", content="Hello knight"),
        ChatMessage(role="assistant", content="Greetings, wanderer."),
    ]
    prompt = build_rp_turn_prompt(
        history=history,
        user_input="What danger lies ahead?",
        authors_note="[Tone: ominous]",
        depth=1,
        max_turns=10,
    )
    assert "USER: Hello knight" in prompt
    assert "ASSISTANT: Greetings, wanderer." in prompt
    assert "SYSTEM: [Tone: ominous]" in prompt
    assert "USER: What danger lies ahead?" in prompt
    assert prompt.endswith("\nASSISTANT:")


def test_extract_agent_response_text_variants():
    assert extract_agent_response_text("Direct reply") == "Direct reply"
    assert extract_agent_response_text(MagicMock(text="  shield!  ")) == "shield!"
    assert extract_agent_response_text(None) == ""


def test_stream_agent_response():
    agent = MagicMock()
    agent.invoke.return_value = "May light guide your step."
    chunks = list(stream_agent_response(agent, "Bless us."))
    assert len(chunks) > 1
    assert "".join(chunks) == "May light guide your step."


def test_stream_agent_response_none():
    agent = MagicMock()
    agent.invoke.return_value = None
    assert list(stream_agent_response(agent, "Bless us.")) == []


