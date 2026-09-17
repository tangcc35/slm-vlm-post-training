import pytest
from google.adk import Workflow
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data, Lorebook, LorebookEntry


def test_agent_registry_instantiates_agent_only_once(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Seraphina",
            description="High Priestess",
            personality="Serene",
            scenario="Temple",
            first_mes="Blessings upon you.",
            mes_example="",
        )
    )
    store.save_character("seraphina", card)

    # First access creates agent
    agent1 = registry.get_or_create_rp_agent("seraphina")
    assert agent1.name == "rp_seraphina"

    # Second access returns the exact same cached instance
    agent2 = registry.get_or_create_rp_agent("seraphina")
    assert agent1 is agent2


def test_agent_registry_character_not_found_raises(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    with pytest.raises(ValueError, match="Character nonexistent not found"):
        registry.get_or_create_rp_agent("nonexistent")


def test_agent_registry_with_explicit_card_and_lorebook(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Gareth",
            description="Knight",
            personality="Brave",
            scenario="Battlefield",
            first_mes="Charge!",
            mes_example="",
        )
    )
    lorebook = Lorebook(name="weapons", entries=[LorebookEntry(keys=["sword"], content="Holy sword")])

    agent = registry.get_or_create_rp_agent("gareth", card=card, lorebook=lorebook)
    assert agent.name == "rp_gareth"
    assert registry.get_or_create_rp_agent("gareth") is agent


def test_agent_registry_register_rp_agent(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    custom_agent = LlmAgent(name="custom_char")
    registry.register_rp_agent("custom_id", custom_agent)

    assert registry.get_or_create_rp_agent("custom_id") is custom_agent


def test_agent_registry_story_workflow_lifecycle(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    wf = Workflow(name="custom_workflow")
    registry.register_story_workflow(wf)
    assert registry.get_story_workflow() is wf
