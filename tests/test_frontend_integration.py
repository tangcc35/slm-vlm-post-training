"""End-to-End System Integration and Verification Test Suite.

Tests full-stack integration across the Dual-Mode Story & Roleplay Workbench:
1. Web UI root `/` mounting with tab markers and frontend CDN dependencies.
2. Static file delivery (`/app.js`, `/style.css`) with appropriate content-types and symbols.
3. Character CRUD lifecycle (create, list, retrieve, delete, 404 handling).
4. Lorebook CRUD lifecycle (create, list, retrieve, delete, 404 handling).
5. Session turn management (inspection, single turn deletion, subsequent truncation/rewind, clearing).
6. Story Co-Pilot expansion (non-streaming) and validation handling.
7. Story Co-Pilot expansion streaming (SSE protocol, delta chunking, full-text event, termination).
8. Roleplay chat and chat streaming integration.
"""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

from story_rp_engine.api.app import create_app
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore


@pytest.fixture
def test_env(tmp_path):
    """Sets up an isolated EngineConfig, EngineStore with InMemorySessionService, and FastAPI app."""
    storage_dir = str(tmp_path / "engine_data")
    config = EngineConfig(storage_dir=storage_dir)
    session_service = InMemorySessionService()
    store = EngineStore(storage_dir=storage_dir, session_service=session_service)
    app = create_app(store=store, config=config)
    client = TestClient(app)
    return {
        "app": app,
        "store": store,
        "config": config,
        "client": client,
        "session_service": session_service,
    }


@pytest.fixture
def client(test_env):
    return test_env["client"]


def test_root_html_serves_all_tabs_and_dependencies(client):
    """Verify root / serves HTML with all 4 tab markers and frontend CDN dependencies."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    html = res.text

    # Verify Page Title and Branding
    assert "Story & Roleplay Workbench" in html
    assert "ADK Dual-Engine Playground" in html

    # Verify all 4 tab markers exist in the HTML template
    assert "activeTab === 'roleplay'" in html
    assert "Roleplay" in html

    assert "activeTab === 'story'" in html
    assert "Story Co-Pilot" in html

    assert "activeTab === 'characters'" in html
    assert "Characters" in html

    assert "activeTab === 'lorebooks'" in html
    assert "Lorebooks" in html

    # Verify CDN script dependencies
    assert "cdn.tailwindcss.com" in html, "Tailwind CSS CDN script missing"
    assert "unpkg.com/vue@3" in html or "vue.global" in html, "Vue 3 CDN script missing"
    assert "unpkg.com/lucide" in html or "lucide" in html, "Lucide CDN script missing"
    assert "marked" in html, "Marked.js CDN script missing"

    # Verify static stylesheet and script links
    assert 'href="style.css"' in html
    assert 'src="app.js?v=5"' in html


def test_static_assets_delivery_and_content_types(client):
    """Verify /app.js and /style.css are served with correct MIME content-types and expected contents."""
    # 1. Verify app.js
    resp_js = client.get("/app.js")
    assert resp_js.status_code == 200
    assert "javascript" in resp_js.headers.get("content-type", "").lower()
    js_content = resp_js.text

    # Key state & methods across domains
    required_js_tokens = [
        # Roleplay workbench
        "rpSessionId",
        "rpCharId",
        "sendRPMessage",
        "rpMessages",
        "deleteTurn",
        "deleteFromHere",
        "regenerateTurn",
        # Story workbench
        "storySessionId",
        "storyPremise",
        "storyMessages",
        "startStory",
        "sendStoryMessage",
        # Character management
        "loadCharacters",
        "saveCharacter",
        "deleteCharacter",
        "charForm",
        # Lorebook management
        "loadLorebooks",
        "saveLorebook",
        "deleteLorebook",
        "lorebookForm",
        # Core helpers
        "renderMarkdown",
        "toast",
    ]
    for token in required_js_tokens:
        assert token in js_content, f"Expected {token} in app.js"

    # 2. Verify style.css
    resp_css = client.get("/style.css")
    assert resp_css.status_code == 200
    assert "css" in resp_css.headers.get("content-type", "").lower()
    css_content = resp_css.text
    assert len(css_content) > 100


def test_character_crud_lifecycle_integration(client):
    """Verify full CRUD lifecycle for character cards via API endpoints."""
    char_id = "valeria_the_blade"
    payload = {
        "char_id": char_id,
        "name": "Valeria the Blade",
        "description": "A seasoned duelist from the northern province.",
        "personality": "Proud, honorable, quick-witted, disciplined.",
        "scenario": "A quiet courtyard at dusk after training.",
        "first_mes": "Hold your guard higher if you intend to challenge me.",
        "mes_example": "<START>\n{{user}}: What style do you fight with?\n{{char}}: The one that keeps me alive.",
        "alternate_greetings": [
            "You have the stance of an amateur.",
            "Care for another bout before the sun goes down?",
        ],
        "tags": ["duelist", "warrior", "north"],
    }

    # 1. Create character
    create_res = client.post("/api/v1/characters", json=payload)
    assert create_res.status_code == 200
    assert create_res.json() == {"status": "saved", "char_id": char_id}

    # 2. List characters
    list_res = client.get("/api/v1/characters")
    assert list_res.status_code == 200
    chars = list_res.json()
    assert char_id in chars
    assert chars[char_id]["name"] == "Valeria the Blade"
    assert chars[char_id]["tags"] == ["duelist", "warrior", "north"]

    # 3. Retrieve character details
    get_res = client.get(f"/api/v1/characters/{char_id}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["name"] == "Valeria the Blade"
    assert detail["personality"] == "Proud, honorable, quick-witted, disciplined."
    assert len(detail["alternate_greetings"]) == 2

    # 4. Delete character
    del_res = client.delete(f"/api/v1/characters/{char_id}")
    assert del_res.status_code == 200
    assert del_res.json() == {"status": "deleted", "char_id": char_id}

    # 5. Verify character is no longer found
    get_after_del = client.get(f"/api/v1/characters/{char_id}")
    assert get_after_del.status_code == 404

    # 6. Second deletion returns 404
    del_again = client.delete(f"/api/v1/characters/{char_id}")
    assert del_again.status_code == 404


def test_lorebook_crud_lifecycle_integration(client):
    """Verify full CRUD lifecycle for lorebooks via API endpoints."""
    lorebook_payload = {
        "name": "Chronicles of Aethelgard",
        "description": "Essential world lore for the fractured realm of Aethelgard.",
        "entries": [
            {
                "keys": ["aethelgard", "realm", "kingdom"],
                "content": "Aethelgard was shattered during the Great Cataclysm of 842.",
                "insertion_order": 10,
                "enabled": True,
            },
            {
                "keys": ["shadow_guild", "underworld"],
                "content": "The Shadow Guild controls subterranean trade routes.",
                "insertion_order": 20,
                "enabled": True,
            },
        ],
    }

    # 1. Create lorebook
    create_res = client.post("/api/v1/lorebooks", json=lorebook_payload)
    assert create_res.status_code == 200
    data = create_res.json()
    assert data["status"] == "saved"
    lb_id = data["lorebook_id"]
    assert lb_id == "chronicles_of_aethelgard"

    # 2. List lorebooks
    list_res = client.get("/api/v1/lorebooks")
    assert list_res.status_code == 200
    lbs = list_res.json()
    assert lb_id in lbs
    assert lbs[lb_id]["name"] == "Chronicles of Aethelgard"

    # 3. Retrieve lorebook detail
    get_res = client.get(f"/api/v1/lorebooks/{lb_id}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["name"] == "Chronicles of Aethelgard"
    assert len(detail["entries"]) == 2
    assert "aethelgard" in detail["entries"][0]["keys"]

    # 4. Delete lorebook
    del_res = client.delete(f"/api/v1/lorebooks/{lb_id}")
    assert del_res.status_code == 200
    assert del_res.json() == {"status": "deleted", "lorebook_id": lb_id}

    # 5. Verify lorebook is no longer found
    get_after_del = client.get(f"/api/v1/lorebooks/{lb_id}")
    assert get_after_del.status_code == 404

    # 6. Second deletion returns 404
    del_again = client.delete(f"/api/v1/lorebooks/{lb_id}")
    assert del_again.status_code == 404


@pytest.mark.anyio
async def test_session_turns_lifecycle_integration(test_env):
    """Verify session turns inspection, single turn deletion, and truncation/rewind."""
    client = test_env["client"]
    session_service = test_env["session_service"]
    session_id = "test_rp_integration_sess"

    # Nonexistent session inspection returns empty turns
    init_res = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert init_res.status_code == 200
    assert init_res.json() == {"turns": []}

    # Populate session with 4 turns
    session = await session_service.create_session(
        app_name="rp_app",
        user_id="User",
        session_id=session_id,
        state={"char_id": "valeria"},
    )
    events = [
        Event(content=types.Content(role="user", parts=[types.Part.from_text(text="Greetings, warrior.")])),
        Event(content=types.Content(role="model", parts=[types.Part.from_text(text="Well met, stranger.")])),
        Event(content=types.Content(role="user", parts=[types.Part.from_text(text="Can we train together?")]),),
        Event(content=types.Content(role="model", parts=[types.Part.from_text(text="Draw your sword and show me what you know.")]),),
    ]
    for ev in events:
        await session_service.append_event(session, ev)

    # 1. Inspect turns
    turns_res = client.get(f"/api/v1/rp/sessions/{session_id}/turns")
    assert turns_res.status_code == 200
    turns = turns_res.json()["turns"]
    assert len(turns) == 4
    assert turns[0] == {"index": 0, "role": "user", "text": "Greetings, warrior."}
    assert turns[1] == {"index": 1, "role": "model", "text": "Well met, stranger."}
    assert turns[2] == {"index": 2, "role": "user", "text": "Can we train together?"}
    assert turns[3] == {"index": 3, "role": "model", "text": "Draw your sword and show me what you know."}

    # 2. Single turn deletion (delete turn at index 2: user question)
    del_turn_res = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 2, "truncate_subsequent": False},
    )
    assert del_turn_res.status_code == 200
    assert del_turn_res.json() == {"status": "ok", "remaining_turns": 3}

    # Verify remaining turns
    turns_after_del = client.get(f"/api/v1/rp/sessions/{session_id}/turns").json()["turns"]
    assert len(turns_after_del) == 3
    assert [t["text"] for t in turns_after_del] == [
        "Greetings, warrior.",
        "Well met, stranger.",
        "Draw your sword and show me what you know.",
    ]

    # 3. Truncate subsequent / rewind (rewind to index 1: keeps index 0 only)
    rewind_res = client.post(
        f"/api/v1/rp/sessions/{session_id}/turns/delete",
        json={"turn_index": 1, "truncate_subsequent": True},
    )
    assert rewind_res.status_code == 200
    assert rewind_res.json() == {"status": "ok", "remaining_turns": 1}

    # Verify remaining turns after rewind
    turns_after_rewind = client.get(f"/api/v1/rp/sessions/{session_id}/turns").json()["turns"]
    assert len(turns_after_rewind) == 1
    assert turns_after_rewind[0]["text"] == "Greetings, warrior."

    # 4. Clear entire session
    clear_res = client.delete(f"/api/v1/rp/sessions/{session_id}")
    assert clear_res.status_code == 200
    assert clear_res.json() == {"status": "deleted", "session_id": session_id}

    turns_after_clear = client.get(f"/api/v1/rp/sessions/{session_id}/turns").json()["turns"]
    assert turns_after_clear == []


def test_story_expand_endpoint_integration(client):
    """Verify non-streaming story expansion endpoint and request validation."""
    session_id = "integration_story_expand_sess"

    # 1. Validation failure on missing session_id
    invalid_res = client.post(
        "/api/v1/story/expand",
        json={
            "premise": "A mysterious spire in the cloud sea.",
        },
    )
    assert invalid_res.status_code == 400
    assert "Missing required fields" in invalid_res.json()["error"]
    assert "session_id" in invalid_res.json()["missing_fields"]

    # 2. Successful expansion
    mock_expansion = "A silhouette emerged from the swirling mists, cloaked in azure silk."

    with patch(
        "story_rp_engine.api.routes_story.execute_runner_turn",
        new_callable=AsyncMock,
        return_value=mock_expansion,
    ) as mock_exec:
        res = client.post(
            "/api/v1/story/expand",
            json={
                "session_id": session_id,
                "premise": "A mysterious spire in the cloud sea.",
                "instruction": "Describe the figure emerging from the mist.",
                "genre": "Fantasy",
                "tone": "Atmospheric",
                "max_tokens": 512,
                "chunk_size": 16,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == session_id
        assert data["expansion"] == mock_expansion
        assert mock_exec.called

        kwargs = mock_exec.call_args.kwargs
        assert kwargs["session_id"] == session_id
        assert kwargs["message"] == "Describe the figure emerging from the mist."
        assert kwargs["state_delta"]["premise"] == "A mysterious spire in the cloud sea."
        assert kwargs["state_delta"]["genre"] == "Fantasy"
        assert kwargs["state_delta"]["tone"] == "Atmospheric"


def test_story_expand_stream_endpoint_integration(client):
    """Verify streaming story expansion endpoint (SSE protocol, chunking, full-text event, DONE marker)."""
    session_id = "integration_story_stream_sess"

    async def mock_stream(*args, **kwargs):
        tokens = ["The", " iron", " gates", " swung", " wide", " with", " a", " groan."]
        for token in tokens:
            yield token

    with patch(
        "story_rp_engine.api.routes_story.stream_runner_turn",
        side_effect=mock_stream,
    ) as mock_stream_fn:
        res = client.post(
            "/api/v1/story/expand/stream",
            json={
                "session_id": session_id,
                "premise": "An abandoned fortress in the winter mountains.",
                "instruction": "Describe the gates opening.",
                "chunk_size": 4,
            },
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert res.headers["x-session-id"] == session_id

        body = res.text
        # Batched 4 tokens then remaining 4 tokens
        assert 'data: {"delta": "The iron gates swung"}\n\n' in body
        assert 'data: {"delta": " wide with a groan."}\n\n' in body
        assert 'data: {"full_text": "The iron gates swung wide with a groan.", "done": true}\n\n' in body
        assert "data: [DONE]\n\n" in body
        assert mock_stream_fn.called


def test_rp_chat_and_stream_endpoints_integration(client):
    """Verify roleplay chat and chat streaming endpoints."""
    char_id = "elena_scout"
    client.post(
        "/api/v1/characters",
        json={
            "char_id": char_id,
            "name": "Elena",
            "description": "A forest scout.",
            "personality": "Observant and quiet.",
            "scenario": "A campsite at twilight.",
            "first_mes": "Keep your voice low.",
        },
    )

    # 1. Non-streaming RP Chat
    with patch(
        "story_rp_engine.api.routes_rp.execute_runner_turn",
        new_callable=AsyncMock,
        return_value="I hear branches snapping in the dark.",
    ) as mock_exec:
        res = client.post(
            "/api/v1/rp/chat",
            json={
                "char_id": char_id,
                "session_id": "rp_test_sess",
                "message": "Did you hear that sound?",
                "user_name": "Ranger",
            },
        )
        assert res.status_code == 200
        assert res.json() == {
            "reply": "I hear branches snapping in the dark.",
            "session_id": "rp_test_sess",
        }
        assert mock_exec.called

    # 2. Streaming RP Chat
    async def mock_stream(*args, **kwargs):
        for token in ["We", " should", " move", " quickly."]:
            yield token

    with patch(
        "story_rp_engine.api.routes_rp.stream_runner_turn",
        side_effect=mock_stream,
    ) as mock_stream_fn:
        stream_res = client.post(
            "/api/v1/rp/chat/stream",
            json={
                "char_id": char_id,
                "session_id": "rp_test_sess",
                "message": "Where are they coming from?",
                "chunk_size": 2,
            },
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        body = stream_res.text
        assert 'data: {"delta": "We should"}\n\n' in body
        assert 'data: {"delta": " move quickly."}\n\n' in body
        assert 'data: {"full_text": "We should move quickly.", "done": true}\n\n' in body
        assert "data: [DONE]\n\n" in body
        assert mock_stream_fn.called


def test_groups_tab_is_served(client):
    html = client.get("/").text
    assert "activeTab === 'groups'" in html
    assert "saveGroup" in html
    js = client.get("/app.js").text
    assert "'/api/v1/groups'" in js
