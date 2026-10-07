from types import SimpleNamespace
import pytest
from google.adk import Workflow
from google.adk.agents import LlmAgent
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.types import CharacterCard, GroupCard, Lorebook, LorebookEntry


@pytest.mark.anyio
async def test_agent_registry_instantiates_agent_only_once(tmp_path):
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
    await store.save_character("seraphina", card)

    # First access creates agent
    agent1 = await registry.get_or_create_rp_agent("seraphina")
    assert agent1.name == "rp_seraphina"

    # Second access returns the exact same cached instance
    agent2 = await registry.get_or_create_rp_agent("seraphina")
    assert agent1 is agent2


@pytest.mark.anyio
async def test_agent_registry_character_not_found_raises(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    with pytest.raises(ValueError, match="Character nonexistent not found"):
        await registry.get_or_create_rp_agent("nonexistent")


@pytest.mark.anyio
async def test_agent_registry_with_explicit_card_and_lorebook(tmp_path):
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

    agent = await registry.get_or_create_rp_agent("gareth", card=card, lorebook=lorebook)
    assert agent.name == "rp_gareth"
    assert await registry.get_or_create_rp_agent("gareth") is agent


@pytest.mark.anyio
async def test_agent_registry_register_rp_agent(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    custom_agent = LlmAgent(name="custom_char")
    registry.register_rp_agent("custom_id", custom_agent)

    assert await registry.get_or_create_rp_agent("custom_id") is custom_agent


def test_agent_registry_story_workflow_lifecycle(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    registry = AgentRegistry(config=config, store=store)

    wf = Workflow(name="custom_workflow")
    registry.register_story_workflow(wf)
    assert registry.get_story_workflow() is wf


@pytest.mark.anyio
async def test_agent_registry_rp_runner_caching(tmp_path):
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
    await store.save_character("seraphina", card)

    runner1 = await registry.get_or_create_rp_runner("seraphina")
    assert runner1 is not None
    assert runner1.app_name == "rp_app"

    # Second call returns the exact same cached runner instance
    runner2 = await registry.get_or_create_rp_runner("seraphina")
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


@pytest.mark.anyio
async def test_agent_registry_register_rp_agent_invalidates_runner(tmp_path):
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
    await store.save_character("seraphina", card)

    runner1 = await registry.get_or_create_rp_runner("seraphina")
    new_agent = LlmAgent(name="rp_seraphina_v2")
    registry.register_rp_agent("seraphina", new_agent)

    runner2 = await registry.get_or_create_rp_runner("seraphina")
    assert runner1 is not runner2
    assert runner2.agent is new_agent


def _seraphina(description: str = "High Priestess") -> CharacterCard:
    return CharacterCard(
        char_id="seraphina",
        name="Seraphina",
        description=description,
        personality="Serene",
        scenario="Temple",
        first_mes="Blessings upon you.",
        mes_example="",
    )


@pytest.mark.anyio
async def test_agent_registry_rebuilds_agent_when_stored_card_changes(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    registry = AgentRegistry(config=EngineConfig(model_name="ollama/llama3.1:8b"), store=store)
    await store.save_character("seraphina", _seraphina())
    agent1 = await registry.get_or_create_rp_agent("seraphina")
    runner1 = await registry.get_or_create_rp_runner("seraphina")

    # Edited through the store, e.g. by another serverless instance sharing the database
    await store.save_character("seraphina", _seraphina(description="Fallen Priestess"))

    runner2 = await registry.get_or_create_rp_runner("seraphina")
    agent2 = await registry.get_or_create_rp_agent("seraphina")
    assert agent2 is not agent1
    assert "Fallen Priestess" in agent2.instruction(SimpleNamespace(state={}))
    assert runner2 is not runner1
    assert runner2.agent is agent2
    assert await registry.get_or_create_rp_runner("seraphina") is runner2


@pytest.mark.anyio
async def test_agent_registry_forgets_character_deleted_from_store(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    registry = AgentRegistry(config=EngineConfig(model_name="ollama/llama3.1:8b"), store=store)
    await store.save_character("seraphina", _seraphina())
    await registry.get_or_create_rp_runner("seraphina")

    await store.delete_character("seraphina")

    with pytest.raises(ValueError, match="Character seraphina not found"):
        await registry.get_or_create_rp_runner("seraphina")
    assert "seraphina" not in registry._rp_agents
    assert "seraphina" not in registry._rp_runners


@pytest.mark.anyio
async def test_agent_registry_register_rp_runner_is_pinned(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    registry = AgentRegistry(config=EngineConfig(model_name="ollama/llama3.1:8b"), store=store)
    await store.save_character("seraphina", _seraphina())
    runner = await registry.get_or_create_rp_runner("seraphina")

    # Registered runners are served as-is, without a store lookup
    registry.register_rp_runner("not_in_store", runner)
    assert await registry.get_or_create_rp_runner("not_in_store") is runner


async def _group_store(tmp_path, char_ids):
    store = EngineStore(storage_dir=str(tmp_path))
    for char_id in ("alice", "bob"):
        await store.save_character(char_id, CharacterCard(char_id=char_id, name=char_id.title()))
    await store.save_group("tavern", GroupCard(group_id="tavern", name="Tavern", char_ids=char_ids))
    return store


@pytest.mark.anyio
async def test_group_runner_cached_and_rebuilt_when_a_member_changes(tmp_path):
    store = await _group_store(tmp_path, ["alice", "bob"])
    registry = AgentRegistry(config=EngineConfig(), store=store)

    runner = await registry.get_or_create_group_runner("tavern")
    assert runner.app_name == "group_app"
    assert await registry.get_or_create_group_runner("tavern") is runner

    await store.save_character("bob", CharacterCard(char_id="bob", name="Bob", description="Now a smith."))
    rebuilt = await registry.get_or_create_group_runner("tavern")
    assert rebuilt is not runner

    registry.forget_group("tavern")
    assert await registry.get_or_create_group_runner("tavern") is not rebuilt


@pytest.mark.anyio
async def test_group_runner_skips_deleted_members(tmp_path, group_models):
    from google.genai import types

    store = await _group_store(tmp_path, ["alice", "ghost"])
    runner = await AgentRegistry(config=EngineConfig(), store=store).get_or_create_group_runner("tavern")

    # The selector picks no one, so every remaining member speaks.
    authors = [
        ev.author
        async for ev in runner.run_async(
            user_id="User", session_id="s1", new_message=types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
        )
    ]
    assert authors == ["speaker_selector", "char_alice"]


@pytest.mark.anyio
async def test_group_runner_errors(tmp_path):
    store = await _group_store(tmp_path, ["ghost"])
    registry = AgentRegistry(config=EngineConfig(), store=store)

    with pytest.raises(ValueError, match="Group missing not found"):
        await registry.get_or_create_group_runner("missing")
    with pytest.raises(ValueError, match="Group tavern has no characters"):
        await registry.get_or_create_group_runner("tavern")
