from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
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
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Lyra",
            "description": "Minstrel",
            "personality": "Cheerful",
            "scenario": "Tavern",
            "first_mes": "Care for a song?",
            "mes_example": "",
        },
    }
    # Create character
    res = client.post("/api/v1/characters?char_id=lyra", json=payload)
    assert res.status_code == 200
    assert res.json() == {"status": "saved", "char_id": "lyra"}

    # Get character
    res = client.get("/api/v1/characters/lyra")
    assert res.status_code == 200
    assert res.json()["data"]["name"] == "Lyra"

    # List characters
    res = client.get("/api/v1/characters")
    assert res.status_code == 200
    assert "lyra" in res.json()


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
        "/api/v1/characters?char_id=lyra",
        json={
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Lyra",
                "description": "A",
                "personality": "B",
                "scenario": "C",
                "first_mes": "D",
                "mes_example": "",
            },
        },
    )

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = "I sing a ballad."

    with patch("story_rp_engine.api.routes_rp.create_rp_agent", return_value=mock_agent):
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
        assert mock_agent.invoke.called

    # Verify history was persisted
    history = store.get_history("session_1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Play something for us."
    assert history[1].role == "assistant"
    assert history[1].content == "I sing a ballad."


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
        "/api/v1/characters?char_id=lyra",
        json={
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Lyra",
                "description": "A",
                "personality": "B",
                "scenario": "C",
                "first_mes": "D",
                "mes_example": "",
            },
        },
    )

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = "Here is a tune."

    with patch("story_rp_engine.api.routes_rp.create_rp_agent", return_value=mock_agent):
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
        assert "data: Here\n\n" in body
        assert "data:  is\n\n" in body
        assert "data:  a\n\n" in body
        assert "data:  tune.\n\n" in body
        assert "data: [DONE]\n\n" in body
        assert mock_agent.invoke.called


    # Verify history was accumulated and persisted
    history = store.get_history("session_2")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Stream a song!"
    assert history[1].role == "assistant"
    assert history[1].content == "Here is a tune."


def test_story_expand_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    mock_writer = MagicMock()
    mock_writer.invoke.return_value = "The ship docked at dawn."

    with patch(
        "story_rp_engine.api.routes_story.prepare_story_expansion",
        return_value=(mock_writer, "writer prompt"),
    ) as mock_prepare:
        res = client.post(
            "/api/v1/story/expand",
            json={
                "premise": "A voyage across the sea.",
                "current_text": "The waves were calm.",
                "instruction": "Describe docking.",
            },
        )
        assert res.status_code == 200
        assert res.json() == {"expansion": "The ship docked at dawn."}
        assert mock_prepare.called
        assert mock_writer.invoke.called


def test_story_expand_stream_endpoint(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    mock_writer = MagicMock()
    mock_writer.invoke.return_value = "The sun rose above the horizon."

    with patch(
        "story_rp_engine.api.routes_story.prepare_story_expansion",
        return_value=(mock_writer, "writer prompt"),
    ) as mock_prepare:
        res = client.post(
            "/api/v1/story/expand/stream",
            json={
                "premise": "Dawn at sea.",
                "current_text": "Morning came.",
                "instruction": "Describe the sun.",
            },
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        body = res.text
        assert "data: The\n\n" in body
        assert "data:  sun\n\n" in body
        assert "data:  rose\n\n" in body
        assert "data:  above\n\n" in body
        assert "data:  the\n\n" in body
        assert "data:  horizon.\n\n" in body
        assert "data: [DONE]\n\n" in body
        assert mock_prepare.called
        assert mock_writer.invoke.called


def test_rp_chat_authors_note_and_custom_user(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    app = create_app(store=store, config=EngineConfig())
    client = TestClient(app)

    client.post(
        "/api/v1/characters?char_id=lyra",
        json={
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Lyra",
                "description": "A",
                "personality": "B",
                "scenario": "C",
                "first_mes": "D",
                "mes_example": "",
            },
        },
    )

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = "Secret chord."

    with patch("story_rp_engine.api.routes_rp.create_rp_agent", return_value=mock_agent) as mock_agent_factory:
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
        assert mock_agent_factory.call_args.kwargs["user_name"] == "Adventurer"
        prompt_arg = mock_agent.invoke.call_args[0][0]
        assert "[Style: Melancholy]" in prompt_arg



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
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Evil",
            "description": "desc",
            "personality": "bad",
            "scenario": "hack",
            "first_mes": "pwn",
            "mes_example": "",
        },
    }

    # Path traversal in save_character query param
    res_save = client.post("/api/v1/characters?char_id=../../evil", json=card_payload)
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
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Valid",
            "description": "a",
            "personality": "b",
            "scenario": "c",
            "first_mes": "d",
            "mes_example": "",
        },
    }
    client.post("/api/v1/characters?char_id=valid_char", json=card_payload)

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
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Valid",
            "description": "a",
            "personality": "b",
            "scenario": "c",
            "first_mes": "d",
            "mes_example": "",
        },
    }
    client.post("/api/v1/characters?char_id=valid_char", json=card_payload)

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

