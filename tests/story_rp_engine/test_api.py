import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, Lorebook, LorebookEntry
from story_rp_engine.storage.store import EngineStore


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_character_endpoints(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    payload = {
        "char_id": "lyra",
        "name": "Lyra",
        "description": "Minstrel",
        "personality": "Cheerful",
        "scenario": "Tavern",
        "first_mes": "Care for a song?",
        "mes_example": "",
    }
    # Create character
    res = client.post("/api/v1/characters", json=payload)
    assert res.status_code == 200
    assert res.json() == {"status": "saved", "char_id": "lyra"}

    # Get character
    res = client.get("/api/v1/characters/lyra")
    assert res.status_code == 200
    assert res.json()["name"] == "Lyra"

    # List characters
    res = client.get("/api/v1/characters")
    assert res.status_code == 200
    assert "lyra" in res.json()


def test_character_save_in_body(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    card_data = {
        "char_id": "elena",
        "name": "Elena",
        "description": "Scholar",
        "personality": "Witty",
        "scenario": "Cavern",
        "first_mes": "Watch your step!",
        "mes_example": "",
    }

    # 1. Flat character card with char_id in body, fixed URL
    res = client.post("/api/v1/characters", json=card_data)
    assert res.status_code == 200
    assert res.json() == {"status": "saved", "char_id": "elena"}

    res_get = client.get("/api/v1/characters/elena")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "Elena"

    # 2. Missing char_id should fail with 400 or 422
    incomplete = {**card_data}
    incomplete.pop("char_id")
    res_missing = client.post("/api/v1/characters", json=incomplete)
    assert res_missing.status_code in (400, 422)

    # 3. Path traversal in char_id inside body rejected with 400
    res_traversal = client.post(
        "/api/v1/characters",
        json={**card_data, "char_id": "../../evil"},
    )
    assert res_traversal.status_code == 400



def test_character_not_found(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    res = client.get("/api/v1/characters/nonexistent")
    assert res.status_code == 404
    assert res.json()["detail"] == "Character not found"


def test_rp_chat_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    # Pre-populate character
    client.post(
        "/api/v1/characters",
        json={
            "char_id": "lyra",
            "name": "Lyra",
            "description": "A",
            "personality": "B",
            "scenario": "C",
            "first_mes": "D",
            "mes_example": "",
        },
    )

    with patch(
        "story_rp_engine.api.routes_rp.execute_runner_turn",
        new_callable=AsyncMock,
        return_value="I sing a ballad.",
    ) as mock_exec:
        res = client.post(
            "/api/v1/rp/chat",
            json={
                "char_id": "lyra",
                "session_id": "session_1",
                "message": "Play something for us.",
            },
        )
        assert res.status_code == 200
        assert res.json() == {"reply": "I sing a ballad.", "session_id": "session_1"}
        assert mock_exec.called


def test_rp_chat_character_not_found(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    res = client.post(
        "/api/v1/rp/chat",
        json={
            "char_id": "ghost",
            "session_id": "session_1",
            "message": "Hello?",
        },
    )
    assert res.status_code == 404
    assert res.json()["detail"] == "Character not found"


def test_rp_chat_stream_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    # Pre-populate character
    client.post(
        "/api/v1/characters",
        json={
            "char_id": "lyra",
            "name": "Lyra",
            "description": "A",
            "personality": "B",
            "scenario": "C",
            "first_mes": "D",
            "mes_example": "",
        },
    )

    async def mock_stream(*args, **kwargs):
        for chunk in ["Here", " is", " a", " tune."]:
            yield chunk

    with patch(
        "story_rp_engine.api.routes_rp.stream_runner_turn",
        side_effect=mock_stream,
    ) as mock_stream_fn:
        res = client.post(
            "/api/v1/rp/chat/stream",
            json={
                "char_id": "lyra",
                "session_id": "session_2",
                "message": "Stream a song!",
            },
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert res.headers["cache-control"] == "no-cache"
        assert res.headers["x-accel-buffering"] == "no"
        body = res.text
        assert 'data: {"delta": "Here is a tune."}\n\n' in body
        assert 'data: {"full_text": "Here is a tune.", "done": true}\n\n' in body
        assert "data: [DONE]\n\n" in body
        assert mock_stream_fn.called


def test_story_expand_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    with patch(
        "story_rp_engine.api.routes_story.execute_runner_turn",
        new_callable=AsyncMock,
        return_value="The gears clicked into place.",
    ) as mock_exec:
        res = client.post(
            "/api/v1/story/expand",
            json={
                "session_id": "story_sess_1",
                "premise": "A clockwork tower.",
                "instruction": "Continue.",
                "genre": "Steampunk",
                "tone": "Dark",
            },
        )
        assert res.status_code == 200
        assert res.json() == {
            "expansion": "The gears clicked into place.",
            "session_id": "story_sess_1",
        }
        assert mock_exec.called
        call_kwargs = mock_exec.call_args.kwargs
        assert call_kwargs["session_id"] == "story_sess_1"
        assert call_kwargs["message"] == "Continue."
        assert call_kwargs["state_delta"]["premise"] == "A clockwork tower."
        assert call_kwargs["state_delta"]["genre"] == "Steampunk"
        assert call_kwargs["state_delta"]["tone"] == "Dark"
        assert call_kwargs["state_delta"]["current_text"] == ""
        assert call_kwargs["state_delta"]["instruction"] == "Continue."


@pytest.mark.anyio
async def test_story_follow_up_uses_end_of_writer_replies(tmp_path):
    from google.adk.events import Event
    from google.genai import types
    from story_rp_engine.api.routes_story import RECENT_TEXT_CHARS

    app = create_app(store=EngineStore(storage_dir=str(tmp_path)), config=EngineConfig())
    service = app.state.agent_registry.get_story_runner().session_service
    session = await service.create_session(app_name="story_app", user_id="User", session_id="s1")
    for author, text in [("story_director", "notes"), ("story_writer", "old " + "x" * RECENT_TEXT_CHARS)]:
        await service.append_event(
            session, Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]))
        )

    with patch("story_rp_engine.api.routes_story.execute_runner_turn", new_callable=AsyncMock, return_value="") as mock_exec:
        res = TestClient(app).post("/api/v1/story/expand", json={"session_id": "s1", "instruction": "Go on."})
    assert res.status_code == 200
    state_delta = mock_exec.call_args.kwargs["state_delta"]
    assert state_delta == {"current_text": "x" * RECENT_TEXT_CHARS, "instruction": "Go on."}
    assert mock_exec.call_args.kwargs["author"] == "story_writer"


@pytest.mark.anyio
async def test_story_lorebook_id_copies_lorebook_into_session(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    lorebook = Lorebook(name="World", entries=[LorebookEntry(keys=["dragon"], content="Dragons sleep under Mount Ash.")])
    await store.save_lorebook("world", lorebook)
    client = TestClient(create_app(store=store, config=EngineConfig()))

    with patch("story_rp_engine.api.routes_story.execute_runner_turn", new_callable=AsyncMock, return_value="") as mock_exec:
        res = client.post("/api/v1/story/expand", json={"session_id": "s1", "lorebook_id": "world"})
        assert res.status_code == 200
        assert mock_exec.call_args.kwargs["state_delta"]["lorebook"] == lorebook.model_dump()

        res = client.post("/api/v1/story/expand", json={"session_id": "s1", "lorebook_id": "missing"})
        assert res.status_code == 404
        assert res.json()["detail"] == "Lorebook not found"


def test_story_expand_endpoint_missing_session_id_fails(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    res = client.post(
        "/api/v1/story/expand",
        json={
            "premise": "A clockwork tower.",
        },
    )
    assert res.status_code == 400
    data = res.json()
    assert "Missing required fields: session_id" in data["error"]
    assert "session_id" in data["missing_fields"]


def test_validation_multiple_missing_fields(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    # CharacterCard requires char_id and name
    res = client.post("/api/v1/characters", json={})
    assert res.status_code == 400
    data = res.json()
    assert "char_id" in data["missing_fields"]
    assert "name" in data["missing_fields"]
    assert "Missing required fields" in data["error"]


def test_story_expand_endpoint_empty_session_id_fails(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    res = client.post(
        "/api/v1/story/expand",
        json={
            "session_id": "   ",
            "premise": "A clockwork tower.",
        },
    )
    assert res.status_code == 400
    assert "session_id is required" in res.json()["detail"].lower()


def test_story_expand_stream_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    async def mock_stream(*args, **kwargs):
        for chunk in ["The", " sun", " rose", " above", " the", " horizon."]:
            yield chunk

    with patch(
        "story_rp_engine.api.routes_story.stream_runner_turn",
        side_effect=mock_stream,
    ) as mock_stream_fn:
        res = client.post(
            "/api/v1/story/expand/stream",
            json={
                "session_id": "story_sess_stream_1",
                "premise": "Dawn at sea.",
                "instruction": "Describe the sun.",
                "chunk_size": 4,
            },
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert res.headers["x-session-id"] == "story_sess_stream_1"
        body = res.text
        # Batched 4 tokens then remaining 2 tokens
        assert 'data: {"delta": "The sun rose above"}\n\n' in body
        assert 'data: {"delta": " the horizon."}\n\n' in body
        assert 'data: {"full_text": "The sun rose above the horizon.", "done": true}\n\n' in body
        assert "data: [DONE]\n\n" in body
        assert mock_stream_fn.called
        call_kwargs = mock_stream_fn.call_args.kwargs
        assert call_kwargs["session_id"] == "story_sess_stream_1"
        assert call_kwargs["message"] == "Describe the sun."
        assert call_kwargs["state_delta"]["premise"] == "Dawn at sea."
        assert call_kwargs["state_delta"]["instruction"] == "Describe the sun."


def test_story_expand_stream_custom_chunk_size(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    async def mock_stream(*args, **kwargs):
        for chunk in ["The", " sun", " rose", " above", " the", " horizon."]:
            yield chunk

    with patch(
        "story_rp_engine.api.routes_story.stream_runner_turn",
        side_effect=mock_stream,
    ):
        res = client.post(
            "/api/v1/story/expand/stream",
            json={
                "session_id": "story_sess_stream_2",
                "instruction": "Describe the sun.",
                "chunk_size": 2,
            },
        )
        assert res.status_code == 200
        assert res.headers["x-session-id"] == "story_sess_stream_2"
        assert res.headers["cache-control"] == "no-cache"
        assert res.headers["x-accel-buffering"] == "no"
        body = res.text
        assert 'data: {"delta": "The sun"}\n\n' in body
        assert 'data: {"delta": " rose above"}\n\n' in body
        assert 'data: {"delta": " the horizon."}\n\n' in body
        assert 'data: {"full_text": "The sun rose above the horizon.", "done": true}\n\n' in body
        assert "data: [DONE]\n\n" in body


def test_rp_chat_authors_note_and_custom_user(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    client.post(
        "/api/v1/characters",
        json={
            "char_id": "lyra",
            "name": "Lyra",
            "description": "A",
            "personality": "B",
            "scenario": "C",
            "first_mes": "D",
            "mes_example": "",
        },
    )

    with patch(
        "story_rp_engine.api.routes_rp.execute_runner_turn",
        new_callable=AsyncMock,
        return_value="Secret chord.",
    ) as mock_exec:
        res = client.post(
            "/api/v1/rp/chat",
            json={
                "char_id": "lyra",
                "session_id": "session_special",
                "message": "Play the hidden song.",
                "authors_note": "[Style: Melancholy]",
                "user_name": "Adventurer",
            },
        )
        assert res.status_code == 200
        assert res.json()["reply"] == "Secret chord."
        assert mock_exec.called
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["user_id"] == "User"
        assert kwargs["session_id"] == "session_special"
        assert kwargs["message"] == "Play the hidden song."
        assert kwargs["state_delta"] == {
            "authors_note": "[Style: Melancholy]",
            "user_name": "Adventurer",
            "greeting": None,
            "char_id": "lyra",
            "last_message": "Play the hidden song.",
        }




def test_cors_headers():
    app = create_app()
    client = TestClient(app)
    res = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert res.headers.get("access-control-allow-credentials") == "true"


def test_default_app_state():
    app = create_app()
    assert isinstance(app.state.store, EngineStore)
    assert isinstance(app.state.config, EngineConfig)


def test_path_traversal_characters_rejected_400(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    card_payload = {
        "char_id": "../../evil",
        "name": "Evil",
        "description": "desc",
        "personality": "bad",
        "scenario": "hack",
        "first_mes": "pwn",
        "mes_example": "",
    }

    # Path traversal in save_character body
    res_save = client.post("/api/v1/characters", json=card_payload)
    assert res_save.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in res_save.json()["detail"]

    # Path traversal in get_character path param
    res_get = client.get("/api/v1/characters/..evil")
    assert res_get.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in res_get.json()["detail"]



def test_path_traversal_rp_chat_rejected_400(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    card_payload = {
        "char_id": "valid_char",
        "name": "Valid",
        "description": "a",
        "personality": "b",
        "scenario": "c",
        "first_mes": "d",
        "mes_example": "",
    }
    client.post("/api/v1/characters", json=card_payload)

    # Bad char_id in chat
    res1 = client.post(
        "/api/v1/rp/chat",
        json={
            "char_id": "../../evil",
            "session_id": "session_valid",
            "message": "hello",
        },
    )
    assert res1.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in res1.json()["detail"]

    # Bad session_id in chat
    res2 = client.post(
        "/api/v1/rp/chat",
        json={
            "char_id": "valid_char",
            "session_id": "../../evil_session",
            "message": "hello",
        },
    )
    assert res2.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in res2.json()["detail"]


def test_path_traversal_rp_chat_stream_rejected_400(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    card_payload = {
        "char_id": "valid_char",
        "name": "Valid",
        "description": "a",
        "personality": "b",
        "scenario": "c",
        "first_mes": "d",
        "mes_example": "",
    }
    client.post("/api/v1/characters", json=card_payload)

    # Bad char_id in stream
    res1 = client.post(
        "/api/v1/rp/chat/stream",
        json={
            "char_id": "../../evil",
            "session_id": "session_valid",
            "message": "hello",
        },
    )
    assert res1.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in res1.json()["detail"]

    # Bad session_id in stream
    res2 = client.post(
        "/api/v1/rp/chat/stream",
        json={
            "char_id": "valid_char",
            "session_id": "../../evil_session",
            "message": "hello",
        },
    )
    assert res2.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in res2.json()["detail"]


@pytest.mark.anyio
async def test_native_runner_chat_execution(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())

    # Pre-register character
    card = CharacterCard(
        char_id="lyra",
        name="Lyra",
        description="A",
        personality="B",
        scenario="C",
        first_mes="D",
        mes_example="",
    )
    await store.save_character("lyra", card)

    # Verify agent is created once in registry and app.state
    assert hasattr(app.state, "agent_registry")
    assert hasattr(app.state, "runner")

    # Verify native execution through ADK BaseLlm model without monkey patching
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class MockLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(
                partial=False,
                content=types.Content(parts=[types.Part.from_text(text="I sing a ballad.")]),
            )

    agent = await app.state.agent_registry.get_or_create_rp_agent("lyra")
    agent.model = MockLlm()

    client = TestClient(app)
    res = client.post(
        "/api/v1/rp/chat",
        json={
            "char_id": "lyra",
            "session_id": "session_native",
            "message": "Sing for me.",
        },
    )
    assert res.status_code == 200
    assert res.json() == {"reply": "I sing a ballad.", "session_id": "session_native"}
    session = await app.state.session_service.get_session(
        app_name="rp_app", user_id="User", session_id="session_native"
    )
    assert session is not None
    assert len(session.events) >= 2
    assert session.events[0].content.parts[0].text == "Sing for me."
    assert session.events[1].content.parts[0].text == "I sing a ballad."


@pytest.mark.anyio
async def test_native_runner_stream_execution(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())

    card = CharacterCard(
        char_id="lyra",
        name="Lyra",
        description="A",
        personality="B",
        scenario="C",
        first_mes="D",
        mes_example="",
    )
    await store.save_character("lyra", card)

    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class MockStreamLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            if stream:
                yield LlmResponse(
                    partial=True,
                    content=types.Content(parts=[types.Part.from_text(text="A lovely")]),
                )
                yield LlmResponse(
                    partial=True,
                    content=types.Content(parts=[types.Part.from_text(text=" song")]),
                )
                yield LlmResponse(
                    partial=False,
                    content=types.Content(parts=[types.Part.from_text(text="A lovely song")]),
                )
            else:
                yield LlmResponse(
                    partial=False,
                    content=types.Content(parts=[types.Part.from_text(text="A lovely song")]),
                )

    agent = await app.state.agent_registry.get_or_create_rp_agent("lyra")
    agent.model = MockStreamLlm()

    client = TestClient(app)
    res = client.post(
        "/api/v1/rp/chat/stream",
        json={
            "char_id": "lyra",
            "session_id": "session_native_stream",
            "message": "Sing live.",
        },
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    body = res.text
    assert 'data: {"delta": "A lovely song"}\n\n' in body
    assert 'data: {"full_text": "A lovely song", "done": true}\n\n' in body
    assert "data: [DONE]\n\n" in body

    session = await app.state.session_service.get_session(
        app_name="rp_app", user_id="User", session_id="session_native_stream"
    )
    assert session is not None
    assert len(session.events) >= 2


@pytest.mark.anyio
async def test_native_runner_story_execution(tmp_path, monkeypatch):
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    captured = {}
    class MockStoryLlm(BaseLlm):
        model: str = "mock"
        prefix: str = ""

        async def generate_content_async(self, llm_request, stream=False):
            captured[self.prefix] = llm_request.config.system_instruction
            yield LlmResponse(
                partial=False,
                content=types.Content(parts=[types.Part.from_text(text=f"{self.prefix} narrative")]),
            )

    from story_rp_engine.story import director_agent, writer_agent
    monkeypatch.setattr(director_agent, "get_adk_model", lambda config: MockStoryLlm(prefix="director"))
    monkeypatch.setattr(writer_agent, "get_adk_model", lambda config: MockStoryLlm(prefix="writer"))
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())

    client = TestClient(app)
    res = client.post(
        "/api/v1/story/expand",
        json={"session_id": "story_sess_inspect", "premise": "A journey north.", "instruction": "Describe the frost."},
    )
    assert res.status_code == 200
    assert res.json()["expansion"] == "writer narrative"
    assert res.json()["session_id"] == "story_sess_inspect"
    assert "Premise: A journey north." in captured["director"]

    # A follow-up chat turn sends only the message: the premise stays in state and the writer's
    # first reply becomes the recent text.
    res = client.post("/api/v1/story/expand", json={"session_id": "story_sess_inspect", "instruction": "Go on."})
    assert res.status_code == 200
    assert "Premise: A journey north." in captured["director"]
    assert "writer narrative" in captured["director"]
    assert "A journey north." in captured["writer"]
    assert "writer narrative" in captured["writer"]
    assert "Go on." in captured["writer"]


def test_story_expand_without_registry_errors(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    app.state.runner = None
    app.state.agent_registry = None
    client = TestClient(app, raise_server_exceptions=False)

    res = client.post(
        "/api/v1/story/expand",
        json={
            "session_id": "test_sess",
            "premise": "A journey north.",
            "instruction": "Describe the frost.",
        },
    )
    assert res.status_code == 500
    assert "agent registry not initialized" in res.json()["detail"].lower()


def test_story_expand_stream_without_registry_errors(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    app.state.runner = None
    app.state.agent_registry = None
    client = TestClient(app, raise_server_exceptions=False)

    res = client.post(
        "/api/v1/story/expand/stream",
        json={
            "session_id": "test_sess",
            "premise": "A journey north.",
            "instruction": "Describe the frost.",
        },
    )
    assert res.status_code == 500
    assert "agent registry not initialized" in res.json()["detail"].lower()


def test_story_expand_multi_turn_stateful(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    session_ids_received = []

    async def mock_execute(runner, user_id, session_id, message, state_delta, author):
        session_ids_received.append((session_id, message, state_delta.get("current_text")))
        return f"Continuation after {message}"

    with patch(
        "story_rp_engine.api.routes_story.execute_runner_turn",
        side_effect=mock_execute,
    ):
        # Turn 1
        res1 = client.post(
            "/api/v1/story/expand",
            json={
                "session_id": "sess_story_chain",
                "premise": "Trapped in the ice.",
                "instruction": "Describe lighting a fire.",
            },
        )
        assert res1.status_code == 200
        assert res1.json()["session_id"] == "sess_story_chain"
        assert res1.json()["expansion"] == "Continuation after Describe lighting a fire."

        # Turn 2: Expanding/modifying using the same session_id
        res2 = client.post(
            "/api/v1/story/expand",
            json={
                "session_id": "sess_story_chain",
                "instruction": "Now make the shadows on the wall shift ominously.",
            },
        )
        assert res2.status_code == 200
        assert res2.json()["session_id"] == "sess_story_chain"
        assert res2.json()["expansion"] == "Continuation after Now make the shadows on the wall shift ominously."

    assert len(session_ids_received) == 2
    assert session_ids_received[0][0] == "sess_story_chain"
    assert session_ids_received[1][0] == "sess_story_chain"


@pytest.mark.anyio
async def test_lifespan_warmup_successful(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())

    with patch("story_rp_engine.api.app.execute_runner_turn", new_callable=AsyncMock) as mock_turn:
        from story_rp_engine.api.app import lifespan
        async with lifespan(app):
            pass
        assert mock_turn.called
        call_kwargs = mock_turn.call_args.kwargs
        assert call_kwargs["user_id"] == "SystemWarmup"
        assert call_kwargs["session_id"] == "warmup_session"


@pytest.mark.anyio
async def test_lifespan_warmup_skipped_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STORY_RP_SKIP_WARMUP", "1")
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())

    with patch("story_rp_engine.api.app.execute_runner_turn", new_callable=AsyncMock) as mock_turn:
        from story_rp_engine.api.app import lifespan
        async with lifespan(app):
            pass
        assert not mock_turn.called


@pytest.mark.anyio
async def test_lifespan_warmup_skipped_for_remote_gemini(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="gemini-flash-latest")
    app = create_app(store=store, config=config)

    with patch("story_rp_engine.api.app.execute_runner_turn", new_callable=AsyncMock) as mock_turn:
        from story_rp_engine.api.app import lifespan
        async with lifespan(app):
            pass
        assert not mock_turn.called


async def _recording_rp_app(tmp_path):
    """App with character 'ava', a lorebook keyed on 'dragon', and recorded RP prompts."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    prompts = []

    class RecordingLlm(BaseLlm):
        model: str = "mock"

        async def generate_content_async(self, llm_request, stream=False):
            texts = [str(llm_request.config.system_instruction)]
            texts += [p.text for c in llm_request.contents for p in c.parts if p.text]
            prompts.append("\n".join(texts))
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text="Hm.")]))

    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig(compaction_enabled=False))
    await store.save_character("ava", CharacterCard(char_id="ava", name="Ava", description="Ava owes {{user}} a favor."))
    await store.save_lorebook(
        "world", Lorebook(name="World", entries=[LorebookEntry(keys=["dragon"], content="Dragons sleep under Mount Ash.")])
    )
    agent = await app.state.agent_registry.get_or_create_rp_agent("ava")
    agent.model = RecordingLlm()
    return TestClient(app), prompts


@pytest.mark.anyio
async def test_rp_chat_lorebook_loaded_once_stays_in_session(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    chat = {"char_id": "ava", "session_id": "s1", "message": "Where is the dragon?"}

    res = client.post("/api/v1/rp/chat/stream", json={**chat, "lorebook_id": "world"})
    assert res.status_code == 200
    assert "Dragons sleep under Mount Ash." in prompts[-1]

    # Later messages omit lorebook_id; the copy loaded into the session is still used.
    res = client.post("/api/v1/rp/chat/stream", json=chat)
    assert res.status_code == 200
    assert "Dragons sleep under Mount Ash." in prompts[-1]


@pytest.mark.anyio
async def test_rp_chat_empty_lorebook_id_clears_session_lorebook(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    chat = {"char_id": "ava", "session_id": "s1", "message": "Where is the dragon?"}

    assert client.post("/api/v1/rp/chat", json={**chat, "lorebook_id": "world"}).status_code == 200
    assert "Dragons sleep under Mount Ash." in prompts[-1]
    assert client.post("/api/v1/rp/chat", json={**chat, "lorebook_id": ""}).status_code == 200
    assert "Dragons sleep under Mount Ash." not in prompts[-1]


@pytest.mark.anyio
async def test_rp_session_list_returns_lorebook_id(tmp_path):
    client, _ = await _recording_rp_app(tmp_path)
    chat = {"char_id": "ava", "session_id": "s1", "message": "Hi"}

    assert client.post("/api/v1/rp/chat", json={**chat, "lorebook_id": "world"}).status_code == 200
    assert client.post("/api/v1/rp/chat", json=chat).status_code == 200
    assert client.get("/api/v1/rp/sessions?char_id=ava").json()[0]["lorebook_id"] == "world"

    assert client.post("/api/v1/rp/chat", json={**chat, "lorebook_id": ""}).status_code == 200
    assert client.get("/api/v1/rp/sessions?char_id=ava").json()[0]["lorebook_id"] is None


@pytest.mark.anyio
async def test_rp_chat_unknown_lorebook_id_returns_404(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    res = client.post(
        "/api/v1/rp/chat/stream",
        json={"char_id": "ava", "session_id": "s1", "message": "Hi", "lorebook_id": "missing"},
    )
    assert res.status_code == 404
    assert res.json()["detail"] == "Lorebook not found"
    assert prompts == []


@pytest.mark.anyio
async def test_rp_chat_greeting_is_in_prompt(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    res = client.post(
        "/api/v1/rp/chat/stream",
        json={"char_id": "ava", "session_id": "s1", "message": "Hello!", "greeting": "Ava waves from the lighthouse door."},
    )
    assert res.status_code == 200
    assert "Ava waves from the lighthouse door." in prompts[-1]


@pytest.mark.anyio
async def test_rp_chat_user_name_fills_user_macro(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    res = client.post("/api/v1/rp/chat", json={"char_id": "ava", "session_id": "s1", "message": "Hi", "user_name": "Alice"})
    assert res.status_code == 200
    assert "Ava owes Alice a favor." in prompts[-1]


@pytest.mark.anyio
async def test_rp_session_turns_found_for_custom_user_name(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    client.post("/api/v1/rp/chat", json={"char_id": "ava", "session_id": "s1", "message": "Hi there", "user_name": "Alice"})
    turns = client.get("/api/v1/rp/sessions/s1/turns").json()["turns"]
    assert [t["text"] for t in turns] == ["Hi there", "Hm."]


@pytest.mark.anyio
async def test_rp_sessions_listed_per_character(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    registry = client.app.state.agent_registry
    await client.app.state.store.save_character("bo", CharacterCard(char_id="bo", name="Bo"))
    (await registry.get_or_create_rp_agent("bo")).model = (await registry.get_or_create_rp_agent("ava")).model
    client.post("/api/v1/rp/chat", json={"char_id": "ava", "session_id": "s1", "message": "First chat", "greeting": "Hi!"})
    client.post("/api/v1/rp/chat", json={"char_id": "ava", "session_id": "s2", "message": "Second chat"})
    client.post("/api/v1/rp/chat", json={"char_id": "bo", "session_id": "s3", "message": "Other character"})

    sessions = client.get("/api/v1/rp/sessions", params={"char_id": "ava"}).json()
    assert [(s["session_id"], s["last_message"]) for s in sessions] == [("s2", "Second chat"), ("s1", "First chat")]
    assert sessions[1]["greeting"] == "Hi!"

    assert client.delete("/api/v1/rp/sessions/s2").status_code == 200
    assert [s["session_id"] for s in client.get("/api/v1/rp/sessions", params={"char_id": "ava"}).json()] == ["s1"]


@pytest.mark.anyio
async def test_story_sessions_list_messages_and_delete(tmp_path):
    from google.adk.events import Event
    from google.genai import types

    app = create_app(store=EngineStore(storage_dir=str(tmp_path)), config=EngineConfig())
    service = app.state.agent_registry.get_story_runner().session_service
    session = await service.create_session(
        app_name="story_app", user_id="User", session_id="s1", state={"premise": "A lighthouse", "genre": "Mystery"}
    )
    for author, text in [("user", "Begin."), ("story_director", "notes"), ("story_writer", "The lamp went dark.")]:
        role = "user" if author == "user" else "model"
        await service.append_event(session, Event(author=author, content=types.Content(role=role, parts=[types.Part(text=text)])))
    client = TestClient(app)

    sessions = client.get("/api/v1/story/sessions").json()
    assert [(s["session_id"], s["premise"]) for s in sessions] == [("s1", "A lighthouse")]

    body = client.get("/api/v1/story/sessions/s1/messages").json()
    assert body["messages"] == [
        {"role": "user", "content": "Begin."},
        {"role": "assistant", "content": "The lamp went dark."},
    ]
    assert body["setup"] == {"premise": "A lighthouse", "genre": "Mystery", "tone": None}

    assert client.delete("/api/v1/story/sessions/s1").status_code == 200
    assert client.get("/api/v1/story/sessions").json() == []
    assert client.get("/api/v1/story/sessions/s1/messages").status_code == 404


@pytest.mark.anyio
async def test_rp_session_rename_keeps_chat_unchanged(tmp_path):
    client, prompts = await _recording_rp_app(tmp_path)
    client.post("/api/v1/rp/chat", json={"char_id": "ava", "session_id": "s1", "message": "Hello"})

    assert client.patch("/api/v1/rp/sessions/s1", json={"title": " Lighthouse night "}).status_code == 200
    assert client.get("/api/v1/rp/sessions", params={"char_id": "ava"}).json()[0]["title"] == "Lighthouse night"
    assert [t["text"] for t in client.get("/api/v1/rp/sessions/s1/turns").json()["turns"]] == ["Hello", "Hm."]

    # The title survives a turn delete, which rebuilds the session.
    client.post("/api/v1/rp/sessions/s1/turns/delete", json={"turn_index": 1})
    assert client.get("/api/v1/rp/sessions", params={"char_id": "ava"}).json()[0]["title"] == "Lighthouse night"
    assert client.patch("/api/v1/rp/sessions/missing", json={"title": "x"}).status_code == 404


@pytest.mark.anyio
async def test_story_session_rename(tmp_path):
    from google.adk.events import Event
    from google.genai import types

    app = create_app(store=EngineStore(storage_dir=str(tmp_path)), config=EngineConfig())
    service = app.state.agent_registry.get_story_runner().session_service
    session = await service.create_session(app_name="story_app", user_id="User", session_id="s1")
    await service.append_event(
        session, Event(author="story_writer", content=types.Content(role="model", parts=[types.Part(text="Once.")]))
    )
    client = TestClient(app)

    assert client.patch("/api/v1/story/sessions/s1", json={"title": "The Lamp"}).status_code == 200
    assert client.get("/api/v1/story/sessions").json()[0]["title"] == "The Lamp"
    assert client.get("/api/v1/story/sessions/s1/messages").json()["messages"] == [{"role": "assistant", "content": "Once."}]
