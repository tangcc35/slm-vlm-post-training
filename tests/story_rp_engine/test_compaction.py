import os
import pytest
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.types import CharacterCard
from story_rp_engine.storage.store import EngineStore


def test_engine_config_compaction_defaults():
    config = EngineConfig()
    assert config.compaction_enabled is True
    assert config.compaction_token_threshold == 4000
    assert config.compaction_event_retention_size == 5
    assert config.compaction_interval == 10
    assert config.compaction_overlap_size == 2
    assert config.compaction_prompt_template is None


def test_engine_config_compaction_env_overrides(monkeypatch):
    monkeypatch.setenv("STORY_RP_COMPACTION_ENABLED", "false")
    monkeypatch.setenv("STORY_RP_COMPACTION_TOKEN_THRESHOLD", "2500")
    monkeypatch.setenv("STORY_RP_COMPACTION_EVENT_RETENTION_SIZE", "3")
    monkeypatch.setenv("STORY_RP_COMPACTION_INTERVAL", "8")
    monkeypatch.setenv("STORY_RP_COMPACTION_OVERLAP_SIZE", "1")
    monkeypatch.setenv("STORY_RP_COMPACTION_PROMPT_TEMPLATE", "Custom template: {conversation_history}")

    config = EngineConfig()
    assert config.compaction_enabled is False
    assert config.compaction_token_threshold == 2500
    assert config.compaction_event_retention_size == 3
    assert config.compaction_interval == 8
    assert config.compaction_overlap_size == 1
    assert config.compaction_prompt_template == "Custom template: {conversation_history}"


@pytest.mark.anyio
async def test_agent_registry_rp_app_compaction_config(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path), db_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db")
    store = EngineStore(storage_dir=str(tmp_path), db_url=config.db_url)
    card = CharacterCard(char_id="hero", name="Hero", description="A brave hero")
    await store.save_character("hero", card)

    registry = AgentRegistry(config, store)
    runner = await registry.get_or_create_rp_runner("hero")

    assert runner.app.events_compaction_config is not None
    cfg = runner.app.events_compaction_config
    assert isinstance(cfg, EventsCompactionConfig)
    assert cfg.token_threshold == 4000
    assert cfg.event_retention_size == 5
    assert cfg.compaction_interval == 10
    assert cfg.overlap_size == 2
    assert isinstance(cfg.summarizer, LlmEventSummarizer)


def test_agent_registry_story_app_compaction_config(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path), db_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db")
    store = EngineStore(storage_dir=str(tmp_path), db_url=config.db_url)

    registry = AgentRegistry(config, store)
    runner = registry.get_story_runner()

    assert runner.app.events_compaction_config is not None
    cfg = runner.app.events_compaction_config
    assert isinstance(cfg, EventsCompactionConfig)
    assert cfg.token_threshold == 4000
    assert cfg.event_retention_size == 5
    assert cfg.compaction_interval == 10
    assert cfg.overlap_size == 2
    assert isinstance(cfg.summarizer, LlmEventSummarizer)


@pytest.mark.anyio
async def test_agent_registry_compaction_disabled(tmp_path):
    config = EngineConfig(
        compaction_enabled=False,
        storage_dir=str(tmp_path),
        db_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db",
    )
    store = EngineStore(storage_dir=str(tmp_path), db_url=config.db_url)
    card = CharacterCard(char_id="hero", name="Hero", description="A brave hero")
    await store.save_character("hero", card)

    registry = AgentRegistry(config, store)
    rp_runner = await registry.get_or_create_rp_runner("hero")
    assert rp_runner.app.events_compaction_config is None

    story_runner = registry.get_story_runner()
    assert story_runner.app.events_compaction_config is None


@pytest.mark.anyio
async def test_session_turns_with_compaction_event(tmp_path):
    from fastapi.testclient import TestClient
    from google.adk.events import Event
    from google.adk.events.event_actions import EventActions, EventCompaction
    from google.genai import types
    from story_rp_engine.api.app import create_app

    config = EngineConfig(storage_dir=str(tmp_path / "engine_data"))
    store = EngineStore(storage_dir=config.storage_dir)
    app = create_app(store=store, config=config)
    client = TestClient(app)

    session = await store.session_service.create_session(
        app_name="rp_app",
        user_id="User",
        session_id="compacted_session",
        state={},
    )
    ev0 = Event(content=types.Content(role="user", parts=[types.Part.from_text(text="Earlier turn")]))
    ev1 = Event(content=types.Content(role="model", parts=[types.Part.from_text(text="Earlier response")]))
    compaction_content = types.Content(role="model", parts=[types.Part.from_text(text="Compacted conversation summary.")])
    ev2 = Event(
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=100.0,
                end_timestamp=200.0,
                compacted_content=compaction_content,
            )
        )
    )
    ev3 = Event(content=types.Content(role="user", parts=[types.Part.from_text(text="New turn")]))

    for ev in [ev0, ev1, ev2, ev3]:
        await store.session_service.append_event(session, ev)

    resp = client.get("/api/v1/rp/sessions/compacted_session/turns")
    assert resp.status_code == 200
    turns = resp.json()["turns"]
    assert len(turns) == 4
    assert turns[2]["role"] == "compaction"
    assert turns[2]["text"] == "Compacted conversation summary."
    assert turns[2]["is_compaction"] is True


class MockCompactionLlm(BaseLlm):
    model: str = "mock-compaction-llm"
    prompt_tokens: int = 100

    async def generate_content_async(self, llm_request, stream=False):
        text = "Assistant reply"
        if "conversation history" in str(llm_request).lower():
            text = "Compacted conversation history summary."
        yield LlmResponse(
            partial=False,
            content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=self.prompt_tokens,
                candidates_token_count=30,
                total_token_count=self.prompt_tokens + 30,
            ),
        )


@pytest.mark.anyio
async def test_sliding_window_compaction_execution(tmp_path):
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    mock = MockCompactionLlm(prompt_tokens=50)
    summarizer = LlmEventSummarizer(llm=mock)
    agent = LlmAgent(name="test_rp_agent", model=mock)
    compaction_cfg = EventsCompactionConfig(
        compaction_interval=2,
        overlap_size=1,
        summarizer=summarizer,
    )
    app = App(name="rp_app", root_agent=agent, events_compaction_config=compaction_cfg)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)

    # Turn 1
    msg1 = types.Content(role="user", parts=[types.Part.from_text(text="First turn")])
    async for _ in runner.run_async(user_id="User", session_id="s_sliding", new_message=msg1):
        pass
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id="s_sliding")
    compaction_events_1 = [e for e in session.events if e.actions and e.actions.compaction]
    assert len(compaction_events_1) == 0

    # Turn 2: triggers sliding window compaction (compaction_interval=2)
    msg2 = types.Content(role="user", parts=[types.Part.from_text(text="Second turn")])
    async for _ in runner.run_async(user_id="User", session_id="s_sliding", new_message=msg2):
        pass
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id="s_sliding")
    compaction_events_2 = [e for e in session.events if e.actions and e.actions.compaction]
    assert len(compaction_events_2) == 1
    comp = compaction_events_2[0].actions.compaction
    assert comp is not None
    assert comp.compacted_content is not None
    assert "Compacted conversation history summary." in comp.compacted_content.parts[0].text


@pytest.mark.anyio
async def test_token_threshold_compaction_execution(tmp_path):
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    # Mock emits prompt_token_count = 5000, exceeding token_threshold=1000
    mock = MockCompactionLlm(prompt_tokens=5000)
    summarizer = LlmEventSummarizer(llm=mock)
    agent = LlmAgent(name="test_token_agent", model=mock)
    compaction_cfg = EventsCompactionConfig(
        token_threshold=1000,
        event_retention_size=1,
        summarizer=summarizer,
    )
    app = App(name="rp_app", root_agent=agent, events_compaction_config=compaction_cfg)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)

    # Execute multiple turns
    msg1 = types.Content(role="user", parts=[types.Part.from_text(text="Turn 1 message")])
    async for _ in runner.run_async(user_id="User", session_id="s_token", new_message=msg1):
        pass
    msg2 = types.Content(role="user", parts=[types.Part.from_text(text="Turn 2 message")])
    async for _ in runner.run_async(user_id="User", session_id="s_token", new_message=msg2):
        pass

    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id="s_token")
    compaction_events = [e for e in session.events if e.actions and e.actions.compaction]
    assert len(compaction_events) >= 1
    comp = compaction_events[0].actions.compaction
    assert comp is not None
    assert comp.compacted_content is not None


@pytest.mark.anyio
async def test_story_workflow_compaction_execution(tmp_path):
    from google.adk import Workflow
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    mock = MockCompactionLlm(prompt_tokens=50)
    summarizer = LlmEventSummarizer(llm=mock)
    dir_agent = LlmAgent(name="director", model=mock)
    wri_agent = LlmAgent(name="writer", model=mock)
    wf = Workflow(name="story_wf", edges=[("START", dir_agent, wri_agent)])

    compaction_cfg = EventsCompactionConfig(
        compaction_interval=2,
        overlap_size=1,
        summarizer=summarizer,
    )
    app = App(name="story_app", root_agent=wf, events_compaction_config=compaction_cfg)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)

    # Turn 1
    async for _ in runner.run_async(
        user_id="User",
        session_id="s_wf",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="Story premise 1")]),
    ):
        pass

    # Turn 2
    async for _ in runner.run_async(
        user_id="User",
        session_id="s_wf",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="Story premise 2")]),
    ):
        pass

    session = await session_service.get_session(app_name="story_app", user_id="User", session_id="s_wf")
    compaction_events = [e for e in session.events if e.actions and e.actions.compaction]
    assert len(compaction_events) >= 1


