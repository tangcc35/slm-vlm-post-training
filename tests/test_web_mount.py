from unittest.mock import patch
from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore


def test_web_ui_mounted():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Story & Roleplay Workbench" in resp.text


def test_web_static_assets():
    app = create_app()
    client = TestClient(app)

    # Verify app.js is served
    resp_js = client.get("/app.js")
    assert resp_js.status_code == 200
    assert "javascript" in resp_js.headers.get("content-type", "").lower()
    # Check that required state and methods are defined in app.js
    required_keywords = [
        "loadCharacters",
        "selectCharacter",
        "saveCharacter",
        "deleteCharacter",
        "importCharacterJSON",
        "exportCharacterJSON",
        "loadLorebooks",
        "selectLorebook",
        "saveLorebook",
        "deleteLorebook",
        "importLorebookJSON",
        "exportLorebookJSON",
        "charForm",
        "lorebookForm",
        "filteredCharacters",
        "filteredLorebooks",
        "checkHealth",
        "toast",
        # Roleplay workbench methods & state (Task 6)
        "rpSessionId",
        "rpCharId",
        "rpUserName",
        "rpAuthorsNote",
        "rpLorebookId",
        "rpChunkSize",
        "rpMessages",
        "selectedGreetingIndex",
        "isGeneratingRP",
        "rpAbortController",
        "onRPCharChange",
        "onGreetingChange",
        "newRPSession",
        "clearRPSession",
        "sendRPMessage",
        "stopGeneratingRP",
        "deleteTurn",
        "deleteFromHere",
        "regenerateTurn",
        "_streamAssistantReply",
        "renderMarkdown",
        "copyMessage",
        # Story Co-Pilot workbench methods & state (Task 7)
        "storySessionId",
        "storyPremise",
        "storyGenre",
        "storyCustomGenre",
        "storyTone",
        "storyCustomTone",
        "storyInstruction",
        "storyMaxTokens",
        "storyChunkSize",
        "storyCurrentText",
        "previousStoryText",
        "isGeneratingStory",
        "storyAbortController",
        "showDirectorBeats",
        "directorBeats",
        "wordCount",
        "estimatedTokens",
        "effectiveGenre",
        "effectiveTone",
        "canUndo",
        "newStorySession",
        "clearStoryText",
        "expandStory",
        "stopGeneratingStory",
        "undoLastExpansion",
        "copyStoryDraft",
        "exportStory",
    ]
    for kw in required_keywords:
        assert kw in resp_js.text, f"Expected {kw} in app.js"

    # Verify style.css is served
    resp_css = client.get("/style.css")
    assert resp_css.status_code == 200
    assert "css" in resp_css.headers.get("content-type", "").lower()


def test_character_and_lorebook_api_flow(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path / "engine_data"))
    store = EngineStore(storage_dir=config.storage_dir)
    app = create_app(store=store, config=config)
    client = TestClient(app)

    # 1. Save character matching app.js payload
    char_payload = {
        "char_id": "elena_test",
        "name": "Elena Test",
        "description": "A rogue scout",
        "personality": "Witty and cautious",
        "scenario": "Forest camp",
        "first_mes": "Keep your voice down.",
        "alternate_greetings": ["Who goes there?"],
        "tags": ["rogue", "fantasy"],
    }
    save_char_resp = client.post("/api/v1/characters", json=char_payload)
    assert save_char_resp.status_code == 200
    assert save_char_resp.json()["status"] == "saved"
    assert save_char_resp.json()["char_id"] == "elena_test"

    # List characters
    list_chars = client.get("/api/v1/characters")
    assert list_chars.status_code == 200
    chars_data = list_chars.json()
    assert "elena_test" in chars_data
    assert chars_data["elena_test"]["name"] == "Elena Test"

    # Get character detail
    get_char = client.get("/api/v1/characters/elena_test")
    assert get_char.status_code == 200
    assert get_char.json()["name"] == "Elena Test"
    assert get_char.json()["alternate_greetings"] == ["Who goes there?"]

    # 2. Save lorebook matching app.js payload
    lb_payload = {
        "name": "Eldoria Realm",
        "description": "High fantasy realm notes",
        "entries": [
            {
                "keys": ["eldoria", "kingdom"],
                "content": "The seat of the high king.",
                "insertion_order": 100,
                "enabled": True,
            }
        ],
    }
    save_lb_resp = client.post("/api/v1/lorebooks", json=lb_payload)
    assert save_lb_resp.status_code == 200
    lb_id = save_lb_resp.json()["lorebook_id"]
    assert lb_id == "eldoria_realm"

    # List lorebooks
    list_lbs = client.get("/api/v1/lorebooks")
    assert list_lbs.status_code == 200
    assert "eldoria_realm" in list_lbs.json()

    # Get lorebook detail
    get_lb = client.get(f"/api/v1/lorebooks/{lb_id}")
    assert get_lb.status_code == 200
    assert get_lb.json()["name"] == "Eldoria Realm"
    assert len(get_lb.json()["entries"]) == 1

    # 3. Clean up / delete
    del_char = client.delete("/api/v1/characters/elena_test")
    assert del_char.status_code == 200
    assert del_char.json()["status"] == "deleted"

    del_lb = client.delete(f"/api/v1/lorebooks/{lb_id}")
    assert del_lb.status_code == 200
    assert del_lb.json()["status"] == "deleted"


def test_rp_session_and_turns_api_flow():
    app = create_app()
    client = TestClient(app)

    session_id = "test_rp_turn_sess"

    # 1. Delete turn on nonexistent session returns ok with 0 remaining turns
    del_turn_resp = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 0, "truncate_subsequent": False},
    )
    assert del_turn_resp.status_code == 200
    assert del_turn_resp.json()["status"] == "ok"

    # 2. Rewind turn on session
    rewind_resp = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 0, "truncate_subsequent": True},
    )
    assert rewind_resp.status_code == 200
    assert rewind_resp.json()["status"] == "ok"

    # 3. Clear session
    clear_resp = client.delete(f"/api/v1/rp/sessions/{session_id}")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "deleted"


def test_story_co_pilot_api_flow(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path / "engine_data"))
    store = EngineStore(storage_dir=config.storage_dir)
    app = create_app(store=store, config=config)
    client = TestClient(app)

    session_id = "test_story_expansion_sess"

    # 1. Expand story non-streaming
    async def mock_execute(runner, user_id, session_id, message, state_delta):
        return "The silver mist drifted silently across the hollow."

    with patch(
        "story_rp_engine.api.routes_story.execute_runner_turn",
        side_effect=mock_execute,
    ):
        res = client.post(
            "/api/v1/story/expand",
            json={
                "session_id": session_id,
                "premise": "A hidden realm at the edge of twilight.",
                "genre": "Fantasy",
                "tone": "Mysterious",
                "current_text": "Night fell over the hills.",
                "instruction": "Describe the eerie mist rising.",
                "max_tokens": 256,
                "chunk_size": 16,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == session_id
        assert "silver mist" in data["expansion"]

    # 2. Expand story streaming
    async def mock_stream(runner, user_id, session_id, message, state_delta):
        yield "The silver "
        yield "mist settled."

    with patch(
        "story_rp_engine.api.routes_story.stream_runner_turn",
        side_effect=mock_stream,
    ):
        stream_res = client.post(
            "/api/v1/story/expand/stream",
            json={
                "session_id": session_id,
                "premise": "A hidden realm at the edge of twilight.",
                "genre": "Fantasy",
                "tone": "Mysterious",
                "current_text": "Night fell over the hills.",
                "instruction": "Describe the eerie mist rising.",
                "max_tokens": 256,
                "chunk_size": 1,
            },
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers.get("content-type", "")
        body = stream_res.text
        assert "data:" in body
        assert "silver" in body
        assert "[DONE]" in body



