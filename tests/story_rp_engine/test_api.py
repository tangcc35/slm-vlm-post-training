import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard
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
                "current_text": "Tick tock.",
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
        assert call_kwargs["state_delta"]["current_text"] == "Tick tock."
        assert call_kwargs["state_delta"]["instruction"] == "Continue."


def test_story_expand_endpoint_missing_session_id_fails(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    res = client.post(
        "/api/v1/story/expand",
        json={
            "premise": "A clockwork tower.",
            "current_text": "Tick tock.",
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
            "current_text": "Tick tock.",
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
                "current_text": "Morning came.",
                "instruction": "Describe the sun.",
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
        assert call_kwargs["state_delta"]["current_text"] == "Morning came."
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
                "current_text": "Morning came.",
                "instruction": "Describe the sun.",
                "chunk_size": 2,
            },
        )
        assert res.status_code == 200
        assert res.headers["x-session-id"] == "story_sess_stream_2"
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
        assert kwargs["user_id"] == "Adventurer"
        assert kwargs["session_id"] == "session_special"
        assert kwargs["message"] == "Play the hidden song."
        assert kwargs["state_delta"] == {"authors_note": "[Style: Melancholy]"}




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
    store.save_character("lyra", card)

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

    agent = app.state.agent_registry.get_or_create_rp_agent("lyra")
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
    store.save_character("lyra", card)

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

    agent = app.state.agent_registry.get_or_create_rp_agent("lyra")
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
async def test_native_runner_story_execution(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())

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

    wf = app.state.agent_registry.get_story_workflow()
    for node in wf.graph.nodes:
        if node.name == "story_director":
            node.model = MockStoryLlm(prefix="director")
        elif node.name == "story_writer":
            node.model = MockStoryLlm(prefix="writer")

    client = TestClient(app)
    res = client.post(
        "/api/v1/story/expand",
        json={
            "session_id": "story_sess_inspect",
            "premise": "A journey north.",
            "current_text": "The wind howled.",
            "instruction": "Describe the frost.",
        },
    )
    assert res.status_code == 200
    assert res.json()["expansion"] == "writer narrative"
    assert res.json()["session_id"] == "story_sess_inspect"
    assert "Premise: A journey north." in captured["director"]
    assert "The wind howled." in captured["director"]
    assert "A journey north." in captured["writer"]
    assert "The wind howled." in captured["writer"]


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
            "current_text": "The wind howled.",
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
            "current_text": "The wind howled.",
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

    async def mock_execute(runner, user_id, session_id, message, state_delta):
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
                "current_text": "The blizzard roared outside.",
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
                "current_text": "The blizzard roared outside.\nThe match caught, sparks flying.",
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




