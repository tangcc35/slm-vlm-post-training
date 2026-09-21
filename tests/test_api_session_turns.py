import pytest
from fastapi.testclient import TestClient
from google.adk.events import Event
from google.genai import types

from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore


@pytest.fixture
def test_setup(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path / "engine_data"))
    store = EngineStore(storage_dir=config.storage_dir)
    app = create_app(store=store, config=config)
    client = TestClient(app)
    return app, store, client


@pytest.fixture
def client(test_setup):
    _, _, client = test_setup
    return client


def test_session_turns_management(client):
    session_id = "test_session_123"

    # Initially empty session or non-existent
    resp = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert resp.status_code == 200
    assert resp.json()["turns"] == []

    # Clear entire session
    del_resp = client.delete(f"/api/v1/rp/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "deleted", "session_id": session_id}


@pytest.mark.anyio
async def test_session_turns_events_and_deletion(test_setup):
    app, store, client = test_setup
    session_id = "test_session_multi"

    # Create session with 4 turns
    session = await store.session_service.create_session(
        app_name="rp_app",
        user_id="User",
        session_id=session_id,
        state={"char_id": "test_hero"},
    )

    ev0 = Event(content=types.Content(role="user", parts=[types.Part.from_text(text="Hello! Who are you?")]))
    ev1 = Event(content=types.Content(role="model", parts=[types.Part.from_text(text="I am a brave knight.")]))
    ev2 = Event(content=types.Content(role="user", parts=[types.Part.from_text(text="Where are we going?")]))
    ev3 = Event(content=types.Content(role="model", parts=[types.Part.from_text(text="To the dragon's lair.")]))

    for ev in [ev0, ev1, ev2, ev3]:
        await store.session_service.append_event(session, ev)

    # 1. Inspect turns
    resp = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["turns"]) == 4
    assert data["turns"][0] == {"index": 0, "role": "user", "text": "Hello! Who are you?"}
    assert data["turns"][1] == {"index": 1, "role": "model", "text": "I am a brave knight."}
    assert data["turns"][2] == {"index": 2, "role": "user", "text": "Where are we going?"}
    assert data["turns"][3] == {"index": 3, "role": "model", "text": "To the dragon's lair."}

    # 2. Delete single turn (e.g. index 2)
    del_turn_resp = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 2, "truncate_subsequent": False},
    )
    assert del_turn_resp.status_code == 200
    assert del_turn_resp.json() == {"status": "ok", "remaining_turns": 3}

    # Verify turns after single deletion
    resp = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert resp.status_code == 200
    remaining_turns = resp.json()["turns"]
    assert len(remaining_turns) == 3
    assert [t["text"] for t in remaining_turns] == [
        "Hello! Who are you?",
        "I am a brave knight.",
        "To the dragon's lair.",
    ]

    # 3. Truncate subsequent turns (rewind to index 1)
    rewind_resp = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 1, "truncate_subsequent": True},
    )
    assert rewind_resp.status_code == 200
    assert rewind_resp.json() == {"status": "ok", "remaining_turns": 1}

    # Verify turns after truncate
    resp = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert resp.status_code == 200
    truncated_turns = resp.json()["turns"]
    assert len(truncated_turns) == 1
    assert truncated_turns[0]["text"] == "Hello! Who are you?"

    # 4. Clear entire session
    clear_resp = client.delete(f"/api/v1/rp/sessions/{session_id}")
    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "deleted", "session_id": session_id}

    # Turns should now be empty
    resp = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert resp.status_code == 200
    assert resp.json()["turns"] == []


def test_session_turns_invalid_session_id(client):
    # Invalid session id / path traversal checks
    bad_id = "..evil"
    resp = client.get(f"/api/v1/rp/sessions/{bad_id}/turns")
    assert resp.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in resp.json()["detail"]

    resp = client.delete(f"/api/v1/rp/sessions/{bad_id}")
    assert resp.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in resp.json()["detail"]

    resp = client.post(
        f"/api/v1/rp/sessions/{bad_id}/turns/delete",
        json={"turn_index": 0, "truncate_subsequent": False},
    )
    assert resp.status_code == 400
    assert "Invalid ID: path traversal characters not allowed" in resp.json()["detail"]


def test_delete_turn_nonexistent_session(client):
    resp = client.post(
        "/api/v1/rp/sessions/nonexistent_session_999/turns/delete",
        json={"turn_index": 0, "truncate_subsequent": False},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "remaining_turns": 0}


@pytest.mark.anyio
async def test_delete_turn_edge_cases(test_setup):
    app, store, client = test_setup
    session_id = "test_session_edges"

    session = await store.session_service.create_session(
        app_name="rp_app",
        user_id="User",
        session_id=session_id,
        state={},
    )
    ev0 = Event(content=types.Content(role="user", parts=[types.Part.from_text(text="Turn 0")]))
    ev1 = Event(content=types.Content(role="model", parts=[types.Part.from_text(text="Turn 1")]))
    await store.session_service.append_event(session, ev0)
    await store.session_service.append_event(session, ev1)

    # 1. Out of bounds single turn deletion (e.g. index 100) -> no-op
    resp = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 100, "truncate_subsequent": False},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "remaining_turns": 2}

    # 2. Negative turn index single turn deletion -> no-op
    resp = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": -5, "truncate_subsequent": False},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "remaining_turns": 2}

    # 3. Truncate all to index 0 -> leaves 0 turns
    resp = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 0, "truncate_subsequent": True},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "remaining_turns": 0}

    # Verify turns is empty
    resp = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert resp.status_code == 200
    assert resp.json()["turns"] == []

