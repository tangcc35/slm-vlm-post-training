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

