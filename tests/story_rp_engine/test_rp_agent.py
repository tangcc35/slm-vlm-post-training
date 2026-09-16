from unittest.mock import MagicMock, patch
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data, ChatMessage, LorebookEntry
from story_rp_engine.rp.agent import create_rp_agent, run_rp_turn


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

    agent = create_rp_agent(card, config, active_lore=[], user_name="Traveler")
    assert agent.name == "rp-theron"
    assert "Theron" in agent.instruction
    assert "Paladin" in agent.instruction
    assert "Traveler" in agent.instruction


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
    lore = [
        LorebookEntry(keys=["silver"], content="Silver Order of Knights"),
    ]

    agent = create_rp_agent(card, config, active_lore=lore, user_name="Traveler")
    assert "Silver Order of Knights" in agent.instruction


def test_run_rp_turn():
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
    agent = create_rp_agent(card, config, active_lore=[], user_name="Traveler")

    with patch.object(agent, "invoke", return_value="I guard the realm."):
        reply = run_rp_turn(
            agent=agent,
            history=[],
            user_input="Who are you?",
            authors_note=None,
        )
        assert reply == "I guard the realm."


def test_run_rp_turn_with_history_and_authors_note():
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
    agent = create_rp_agent(card, config, active_lore=[], user_name="Traveler")

    history = [
        ChatMessage(role="user", content="Hello knight"),
        ChatMessage(role="assistant", content="Greetings, wanderer."),
    ]

    mock_invoke = MagicMock(return_value="The storm approaches.")
    with patch.object(agent, "invoke", mock_invoke):
        reply = run_rp_turn(
            agent=agent,
            history=history,
            user_input="What danger lies ahead?",
            authors_note="[Tone: ominous]",
            depth=1,
            max_turns=10,
        )
        assert reply == "The storm approaches."
        mock_invoke.assert_called_once()
        prompt_arg = mock_invoke.call_args[0][0]
        assert "USER: Hello knight" in prompt_arg
        assert "ASSISTANT: Greetings, wanderer." in prompt_arg
        assert "SYSTEM: [Tone: ominous]" in prompt_arg
        assert "USER: What danger lies ahead?" in prompt_arg
        assert prompt_arg.endswith("\nASSISTANT:")


def test_run_rp_turn_response_object():
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
    agent = create_rp_agent(card, config, active_lore=[], user_name="Traveler")

    response_obj = MagicMock()
    response_obj.text = "   Stand firm behind my shield!   "

    with patch.object(agent, "invoke", return_value=response_obj):
        reply = run_rp_turn(
            agent=agent,
            history=[],
            user_input="Stand with me!",
        )
        assert reply == "Stand firm behind my shield!"
