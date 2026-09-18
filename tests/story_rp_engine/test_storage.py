import pytest
from google.adk.sessions import DatabaseSessionService, Session
from story_rp_engine.core.types import (
    CharacterCardV2,
    CharacterCardV2Data,
    Lorebook,
    LorebookEntry,
)
from story_rp_engine.storage.store import EngineStore


def test_character_crud(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Valerie",
            description="Alchemist",
            personality="Witty",
            scenario="Shop",
            first_mes="Can I help you?",
            mes_example="",
        )
    )
    store.save_character("valerie_1", card)

    retrieved = store.get_character("valerie_1")
    assert retrieved is not None
    assert retrieved.data.name == "Valerie"

    all_chars = store.list_characters()
    assert "valerie_1" in all_chars
    assert all_chars["valerie_1"].data.name == "Valerie"


def test_character_not_found(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    assert store.get_character("nonexistent") is None


def test_lorebook_crud(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    lorebook = Lorebook(
        name="Arcane Lore",
        description="Lore regarding magic",
        entries=[
            LorebookEntry(
                keys=["elixir", "potion"],
                content="Alchemical drafts with restorative properties.",
                insertion_order=10,
            )
        ],
    )
    store.save_lorebook("arcane_lore", lorebook)

    retrieved = store.get_lorebook("arcane_lore")
    assert retrieved is not None
    assert retrieved.name == "Arcane Lore"
    assert len(retrieved.entries) == 1
    assert retrieved.entries[0].keys == ["elixir", "potion"]

    all_lore = store.list_lorebooks()
    assert "arcane_lore" in all_lore
    assert all_lore["arcane_lore"].name == "Arcane Lore"

    assert store.get_lorebook("nonexistent_lore") is None


def test_list_ignores_non_json_files(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    # Write dummy non-json files
    (tmp_path / "characters" / "notes.txt").write_text("not a json")
    (tmp_path / "lorebooks" / "README.md").write_text("some docs")

    assert store.list_characters() == {}
    assert store.list_lorebooks() == {}


@pytest.mark.parametrize(
    "bad_key",
    [
        "../../evil",
        r"..\..\evil",
        "nested/key",
        r"nested\key",
        "..",
        "../something",
        "   ../../evil   ",
    ],
)
def test_path_traversal_defense(tmp_path, bad_key):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Valerie",
            description="Alchemist",
            personality="Witty",
            scenario="Shop",
            first_mes="Can I help you?",
            mes_example="",
        )
    )
    lorebook = Lorebook(name="Arcane", description="Lore", entries=[])

    # Verify all CRUD methods reject path traversal payloads
    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        store.save_character(bad_key, card)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        store.get_character(bad_key)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        store.save_lorebook(bad_key, lorebook)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        store.get_lorebook(bad_key)


def test_key_whitespace_stripping(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Valerie",
            description="Alchemist",
            personality="Witty",
            scenario="Shop",
            first_mes="Can I help you?",
            mes_example="",
        )
    )
    store.save_character("  valerie_padded  ", card)
    retrieved = store.get_character("valerie_padded")
    assert retrieved is not None
    assert retrieved.data.name == "Valerie"


@pytest.mark.anyio
async def test_store_session_service_lifecycle(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    session = await store.get_or_create_session(
        app_name="rp_app",
        user_id="user_123",
        session_id="session_abc",
        initial_state={"char_id": "lyra"},
    )
    assert isinstance(session, Session)
    assert session.id == "session_abc"
    assert session.state.get("char_id") == "lyra"

    # Re-fetching returns the existing session
    session_again = await store.get_or_create_session(
        app_name="rp_app",
        user_id="user_123",
        session_id="session_abc",
    )
    assert session_again.id == "session_abc"
    assert session_again.state.get("char_id") == "lyra"


@pytest.mark.anyio
async def test_database_session_service_default(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    assert isinstance(store.session_service, DatabaseSessionService)
    session = await store.get_or_create_session(
        app_name="test_app", user_id="User", session_id="sess_1"
    )
    assert session.id == "sess_1"
    fetched = await store.session_service.get_session(
        app_name="test_app", user_id="User", session_id="sess_1"
    )
    assert fetched is not None
    assert fetched.id == "sess_1"


@pytest.mark.anyio
async def test_database_session_service_custom_url(tmp_path):
    db_file = tmp_path / "custom.db"
    store = EngineStore(
        storage_dir=str(tmp_path),
        db_url=f"sqlite+aiosqlite:///{db_file}",
    )
    assert isinstance(store.session_service, DatabaseSessionService)
    session = await store.get_or_create_session(
        app_name="test_app", user_id="User", session_id="sess_custom"
    )
    assert session.id == "sess_custom"
