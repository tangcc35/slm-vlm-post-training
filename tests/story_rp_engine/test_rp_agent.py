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
    assert "Theron" in agent.instruction
    assert "Paladin" in agent.instruction
    assert "Traveler" in agent.instruction
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
    assert "Silver Order of Knights" not in agent.instruction


