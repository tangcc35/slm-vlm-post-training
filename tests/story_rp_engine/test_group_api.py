import pytest
from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, Lorebook, LorebookEntry
from story_rp_engine.storage.store import EngineStore

TAVERN = {
    "group_id": "tavern",
    "name": "Tavern",
    "char_ids": ["alice", "bob"],
    "scenario": "A rainy night.",
}


async def _group_client(tmp_path):
    """App with characters alice and bob, the group 'tavern', and a lorebook keyed on 'forge'."""
    store = EngineStore(storage_dir=str(tmp_path))
    for char_id, name in [("alice", "Alice"), ("bob", "Bob")]:
        await store.save_character(char_id, CharacterCard(char_id=char_id, name=name))
    await store.save_lorebook(
        "world", Lorebook(name="World", entries=[LorebookEntry(keys=["forge"], content="The forge never cools.")])
    )
    client = TestClient(create_app(store=store, config=EngineConfig(compaction_enabled=False)))
    assert client.post("/api/v1/groups", json=TAVERN).status_code == 200
    return client


def _chat(client, message, **extra):
    body = {"group_id": "tavern", "session_id": "s1", "message": message, "chunk_size": 1, **extra}
    return client.post("/api/v1/group/chat/stream", json=body)


def test_group_crud(tmp_path):
    client = TestClient(create_app(store=EngineStore(storage_dir=str(tmp_path)), config=EngineConfig()))

    assert client.post("/api/v1/groups", json=TAVERN).json() == {"status": "saved", "group_id": "tavern"}
    assert client.get("/api/v1/groups/tavern").json()["char_ids"] == ["alice", "bob"]
    assert list(client.get("/api/v1/groups").json()) == ["tavern"]
    assert client.post("/api/v1/groups", json={**TAVERN, "group_id": "../evil"}).status_code == 400

    assert client.delete("/api/v1/groups/tavern").json() == {"status": "deleted", "group_id": "tavern"}
    assert client.get("/api/v1/groups/tavern").status_code == 404
    assert client.delete("/api/v1/groups/tavern").status_code == 404


@pytest.mark.anyio
async def test_group_chat_streams_replies_by_speaker(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["bob", "alice"]}

    res = _chat(client, "Hello!")
    assert res.status_code == 200
    assert res.text == (
        'data: {"speaker": "bob", "delta": "Bob "}\n\n'
        'data: {"speaker": "bob", "delta": "line"}\n\n'
        'data: {"speaker": "alice", "delta": "Alice "}\n\n'
        'data: {"speaker": "alice", "delta": "line"}\n\n'
        'data: {"replies": [{"speaker": "bob", "text": "Bob line"}, {"speaker": "alice", "text": "Alice line"}], "done": true}\n\n'
        "data: [DONE]\n\n"
    )


@pytest.mark.anyio
async def test_group_chat_loads_lorebook_and_scenario(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["alice"]}

    assert _chat(client, "Is the forge lit?", lorebook_id="world").status_code == 200
    request = group_models.requests["Alice"][-1]
    assert "The forge never cools." in "".join(p.text for p in request.contents[-1].parts)
    assert "A rainy night." in request.config.system_instruction


@pytest.mark.anyio
async def test_group_turns_name_speakers_and_hide_the_selector(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["bob", "alice"]}
    _chat(client, "Hello!")

    turns = client.get("/api/v1/group/sessions/s1/turns").json()["turns"]
    assert [(t["speaker"], t["text"]) for t in turns] == [(None, "Hello!"), ("bob", "Bob line"), ("alice", "Alice line")]

    sessions = client.get("/api/v1/group/sessions", params={"group_id": "tavern"}).json()
    assert [(s["session_id"], s["last_message"]) for s in sessions] == [("s1", "Hello!")]
    assert client.get("/api/v1/group/sessions", params={"group_id": "other"}).json() == []


@pytest.mark.anyio
async def test_group_session_rename_delete_turn_and_delete(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["bob", "alice"]}
    _chat(client, "Hello!")

    assert client.patch("/api/v1/group/sessions/s1", json={"title": "Rainy night"}).status_code == 200
    # Index 1 is Bob's reply: the selector's hidden event doesn't count.
    client.post("/api/v1/group/sessions/s1/turns/delete", json={"turn_index": 1})
    assert [t["text"] for t in client.get("/api/v1/group/sessions/s1/turns").json()["turns"]] == ["Hello!", "Alice line"]
    assert client.get("/api/v1/group/sessions", params={"group_id": "tavern"}).json()[0]["title"] == "Rainy night"

    assert client.delete("/api/v1/group/sessions/s1").status_code == 200
    assert client.get("/api/v1/group/sessions", params={"group_id": "tavern"}).json() == []


@pytest.mark.anyio
async def test_group_chat_rejects_bad_requests(tmp_path, group_models):
    client = await _group_client(tmp_path)
    client.post("/api/v1/groups", json={**TAVERN, "group_id": "ghosts", "char_ids": ["ghost"]})

    assert _chat(client, "Hi", group_id="missing").status_code == 404
    res = _chat(client, "Hi", group_id="ghosts")
    assert res.status_code == 400
    assert "has no characters" in res.json()["detail"]
    assert _chat(client, "Hi", session_id="../evil").status_code == 400
    assert _chat(client, "Hi", lorebook_id="missing").status_code == 404


@pytest.mark.anyio
async def test_group_chat_reports_unusable_selector_reply(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = "not json"

    res = _chat(client, "Hello!")
    assert res.status_code == 200
    assert 'data: {"error": ' in res.text
    assert res.text.endswith("data: [DONE]\n\n")


@pytest.mark.anyio
async def test_characters_never_see_selector_json_across_persisted_turns(tmp_path, group_models):
    client = await _group_client(tmp_path)
    group_models.plan = {"speakers": ["bob", "alice"]}
    _chat(client, "Hello!")
    _chat(client, "And now?")

    texts = ["".join(p.text or "" for p in c.parts) for c in group_models.requests["Alice"][-1].contents]
    assert any("[char_bob] said:" in t for t in texts)
    assert not any('"speakers"' in t for t in texts)
