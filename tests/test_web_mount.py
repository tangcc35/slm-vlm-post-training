from fastapi.testclient import TestClient
from story_rp_engine.api.app import create_app

def test_web_ui_mounted():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Story & Roleplay Workbench" in resp.text
