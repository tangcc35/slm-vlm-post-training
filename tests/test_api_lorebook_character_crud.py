import pytest
from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.config import EngineConfig


@pytest.fixture
def client(tmp_path):
    config = EngineConfig(storage_dir=str(tmp_path / "engine_data"))
    store = EngineStore(storage_dir=config.storage_dir)
    app = create_app(store=store, config=config)
    return TestClient(app)


def test_character_deletion(client):
    card_data = {
        "char_id": "test_char",
        "name": "Test Character",
        "description": "A character for testing deletion",
    }
    create_resp = client.post("/api/v1/characters", json=card_data)
    assert create_resp.status_code == 200

    del_resp = client.delete("/api/v1/characters/test_char")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "deleted", "char_id": "test_char"}

    get_resp = client.get("/api/v1/characters/test_char")
    assert get_resp.status_code == 404

    # Deleting again should return 404
    del_again = client.delete("/api/v1/characters/test_char")
    assert del_again.status_code == 404


def test_lorebook_crud(client):
    lb_data = {
        "name": "Eldoria World",
        "description": "World lore for Eldoria",
        "entries": [
            {
                "keys": ["eldoria", "kingdom"],
                "content": "A high fantasy kingdom.",
                "insertion_order": 100,
                "enabled": True,
            }
        ],
    }
    save_resp = client.post("/api/v1/lorebooks", json=lb_data)
    assert save_resp.status_code == 200
    lb_id = save_resp.json()["lorebook_id"]

    get_resp = client.get(f"/api/v1/lorebooks/{lb_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Eldoria World"

    list_resp = client.get("/api/v1/lorebooks")
    assert list_resp.status_code == 200
    assert lb_id in list_resp.json()

    del_resp = client.delete(f"/api/v1/lorebooks/{lb_id}")
    assert del_resp.status_code == 200

    get_after_del = client.get(f"/api/v1/lorebooks/{lb_id}")
    assert get_after_del.status_code == 404

    # Deleting again should return 404
    del_again = client.delete(f"/api/v1/lorebooks/{lb_id}")
    assert del_again.status_code == 404
