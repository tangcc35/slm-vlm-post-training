from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore


def test_full_engine_lifecycle(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    app = create_app(store=store, config=config)
    client = TestClient(app)

    # 1. Health check
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # 2. Register character card
    card_json = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Rowan",
            "description": "A seasoned frontier guide.",
            "personality": "Gruff but reliable.",
            "scenario": "A mountain pass in winter.",
            "first_mes": "Pack your gear tight.",
            "mes_example": "{{user}}: How far?\n{{char}}: Two days, if the snow holds.",
        },
    }
    save_res = client.post("/api/v1/characters?char_id=rowan", json=card_json)
    assert save_res.status_code == 200

    # 3. Conversational RP turn
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = "Snow is coming. Move faster."
    with patch("story_rp_engine.api.routes_rp.create_rp_agent", return_value=mock_agent):
        chat_res = client.post(
            "/api/v1/rp/chat",
            json={
                "char_id": "rowan",
                "session_id": "sess_mountain_1",
                "message": "Do you smell snow?",
            },
        )
        assert chat_res.status_code == 200
        assert "Snow is coming" in chat_res.json()["reply"]


    # 4. Verify session persisted
    saved_history = store.get_history("sess_mountain_1")
    assert len(saved_history) == 2
    assert saved_history[0].content == "Do you smell snow?"
    assert saved_history[1].content == "Snow is coming. Move faster."

    # 5. Story Co-Pilot expansion
    mock_writer = MagicMock()
    mock_writer.invoke.return_value = "The ridge gave way to a vast frozen valley."
    with patch(
        "story_rp_engine.api.routes_story.prepare_story_expansion",
        return_value=(mock_writer, "writer prompt"),
    ):
        story_res = client.post(
            "/api/v1/story/expand",
            json={
                "premise": "Surviving the high winter pass.",
                "current_text": "Rowan tightened the straps on his pack.",
                "instruction": "Describe the view from the pass.",
                "genre": "Adventure",
                "tone": "Gritty",
            },
        )
        assert story_res.status_code == 200
        assert "frozen valley" in story_res.json()["expansion"]
