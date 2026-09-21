import pytest
from google.adk import Workflow
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.types import CharacterCard, Lorebook, LorebookEntry


def test_agent_registry_instantiates_agent_only_once(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    card = CharacterCard(
        char_id="seraphina",
        name="Seraphina",
        description="High Priestess",
        personality="Serene",
        scenario="Temple",
        first_mes="Blessings upon you.",
        mes_example="",
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

    card = CharacterCard(
        char_id="gareth",
        name="Gareth",
        description="Knight",
        personality="Brave",
        scenario="Battlefield",
        first_mes="Charge!",
        mes_example="",
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


def test_agent_registry_rp_runner_caching(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    card = CharacterCard(
        char_id="seraphina",
        name="Seraphina",
        description="High Priestess",
        personality="Serene",
        scenario="Temple",
        first_mes="Blessings upon you.",
        mes_example="",
    )
    store.save_character("seraphina", card)

    runner1 = registry.get_or_create_rp_runner("seraphina")
    assert runner1 is not None
    assert runner1.app_name == "rp_app"

    # Second call returns the exact same cached runner instance
    runner2 = registry.get_or_create_rp_runner("seraphina")
    assert runner1 is runner2


def test_agent_registry_story_runner_caching(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    runner1 = registry.get_story_runner()
    assert runner1 is not None
    assert runner1.app_name == "story_app"

    # Second call returns the exact same cached runner instance
    runner2 = registry.get_story_runner()
    assert runner1 is runner2


def test_agent_registry_register_rp_agent_invalidates_runner(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    card = CharacterCard(
        char_id="seraphina",
        name="Seraphina",
        description="Priestess",
        personality="Serene",
        scenario="Temple",
        first_mes="Blessings.",
        mes_example="",
    )
    store.save_character("seraphina", card)

    runner1 = registry.get_or_create_rp_runner("seraphina")
    new_agent = LlmAgent(name="rp_seraphina_v2")
    registry.register_rp_agent("seraphina", new_agent)

    runner2 = registry.get_or_create_rp_runner("seraphina")
    assert runner1 is not runner2
    assert runner2.agent is new_agent
