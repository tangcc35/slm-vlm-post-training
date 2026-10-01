import pytest
from google.adk.sessions import DatabaseSessionService, Session
from story_rp_engine.core.types import (
    CharacterCard,
    Lorebook,
    LorebookEntry,
)
from story_rp_engine.storage.store import EngineStore


@pytest.mark.anyio
async def test_character_crud(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCard(
        char_id="valerie_1",
        name="Valerie",
        description="Alchemist",
        personality="Witty",
        scenario="Shop",
        first_mes="Can I help you?",
        mes_example="",
    )
    await store.save_character("valerie_1", card)

    retrieved = await store.get_character("valerie_1")
    assert retrieved is not None
    assert retrieved.name == "Valerie"

    all_chars = await store.list_characters()
    assert "valerie_1" in all_chars
    assert all_chars["valerie_1"].name == "Valerie"


def test_character_crud_sync(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCard(
        char_id="valerie_sync",
        name="Valerie Sync",
        description="Alchemist",
        personality="Witty",
        scenario="Shop",
        first_mes="Can I help you?",
        mes_example="",
    )
    store.save_character_sync("valerie_sync", card)
    retrieved = store.get_character_sync("valerie_sync")
    assert retrieved is not None
    assert retrieved.name == "Valerie Sync"

    all_chars = store.list_characters_sync()
    assert "valerie_sync" in all_chars

    store.delete_character_sync("valerie_sync")
    assert store.get_character_sync("valerie_sync") is None


@pytest.mark.anyio
async def test_character_not_found(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    assert await store.get_character("nonexistent") is None


@pytest.mark.anyio
async def test_lorebook_crud(tmp_path):
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
    await store.save_lorebook("arcane_lore", lorebook)

    retrieved = await store.get_lorebook("arcane_lore")
    assert retrieved is not None
    assert retrieved.name == "Arcane Lore"
    assert len(retrieved.entries) == 1
    assert retrieved.entries[0].keys == ["elixir", "potion"]

    all_lore = await store.list_lorebooks()
    assert "arcane_lore" in all_lore
    assert all_lore["arcane_lore"].name == "Arcane Lore"

    assert await store.get_lorebook("nonexistent_lore") is None


def test_lorebook_crud_sync(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    lorebook = Lorebook(
        name="Arcane Lore",
        description="Lore regarding magic",
        entries=[],
    )
    store.save_lorebook_sync("arcane_sync", lorebook)
    assert store.get_lorebook_sync("arcane_sync") is not None
    all_lore = store.list_lorebooks_sync()
    assert "arcane_sync" in all_lore
    store.delete_lorebook_sync("arcane_sync")
    assert store.get_lorebook_sync("arcane_sync") is None


@pytest.mark.anyio
async def test_list_ignores_non_json_files(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    # Write dummy non-json files
    (tmp_path / "characters" / "notes.txt").write_text("not a json")
    (tmp_path / "lorebooks" / "README.md").write_text("some docs")

    assert await store.list_characters() == {}
    assert await store.list_lorebooks() == {}


@pytest.mark.anyio
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
async def test_path_traversal_defense(tmp_path, bad_key):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCard(
        char_id="valerie",
        name="Valerie",
        description="Alchemist",
        personality="Witty",
        scenario="Shop",
        first_mes="Can I help you?",
        mes_example="",
    )
    lorebook = Lorebook(name="Arcane", description="Lore", entries=[])

    # Verify all CRUD methods reject path traversal payloads
    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.save_character(bad_key, card)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.get_character(bad_key)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.save_lorebook(bad_key, lorebook)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.get_lorebook(bad_key)


@pytest.mark.anyio
async def test_key_whitespace_stripping(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCard(
        char_id="valerie",
        name="Valerie",
        description="Alchemist",
        personality="Witty",
        scenario="Shop",
        first_mes="Can I help you?",
        mes_example="",
    )
    await store.save_character("  valerie_padded  ", card)
    retrieved = await store.get_character("valerie_padded")
    assert retrieved is not None
    assert retrieved.name == "Valerie"


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
    assert (tmp_path / "sessions.db").is_file()


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
    assert db_file.is_file()


@pytest.mark.anyio
async def test_character_database_crud_and_serverless_restart(tmp_path):
    db_file = tmp_path / "shared.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    # Instance 1: Saves a character with empty /tmp storage dir 1
    dir1 = tmp_path / "instance_1"
    store1 = EngineStore(storage_dir=str(dir1), db_url=db_url)

    card = CharacterCard(
        char_id="cloud_valerie",
        name="Valerie in the Cloud",
        description="Alchemist with cloud persistence",
        personality="Resourceful",
        scenario="Neon DB",
        first_mes="Hello from the database!",
    )
    await store1.save_character(card.char_id, card)

    retrieved1 = await store1.get_character("cloud_valerie")
    assert retrieved1 is not None
    assert retrieved1.name == "Valerie in the Cloud"

    # Instance 2 (Cold start simulation): New container with empty /tmp storage dir 2
    dir2 = tmp_path / "instance_2"
    store2 = EngineStore(storage_dir=str(dir2), db_url=db_url)

    # Instance 2 has no local files in dir2/characters
    assert not (dir2 / "characters" / "cloud_valerie.json").exists()

    # But store2 can retrieve the character from the shared database!
    retrieved2 = await store2.get_character("cloud_valerie")
    assert retrieved2 is not None
    assert retrieved2.name == "Valerie in the Cloud"

    all_chars = await store2.list_characters()
    assert "cloud_valerie" in all_chars

    # Deletion test
    await store2.delete_character("cloud_valerie")
    assert await store2.get_character("cloud_valerie") is None

    # Verify deleted on store1 as well
    store1._character_cache.clear()
    assert await store1.get_character("cloud_valerie") is None


@pytest.mark.anyio
async def test_lorebook_database_crud_and_serverless_restart(tmp_path):
    db_file = tmp_path / "shared_lore.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    # Instance 1 saves lorebook
    dir1 = tmp_path / "instance_1"
    store1 = EngineStore(storage_dir=str(dir1), db_url=db_url)

    lorebook = Lorebook(
        name="Database Lore",
        description="Lore persisted in cloud DB",
        entries=[
            LorebookEntry(
                keys=["neon", "postgres"],
                content="Serverless Postgres database.",
            )
        ],
    )
    await store1.save_lorebook("db_lore", lorebook)

    # Instance 2 (Cold start simulation)
    dir2 = tmp_path / "instance_2"
    store2 = EngineStore(storage_dir=str(dir2), db_url=db_url)

    retrieved = await store2.get_lorebook("db_lore")
    assert retrieved is not None
    assert retrieved.name == "Database Lore"
    assert len(retrieved.entries) == 1
    assert retrieved.entries[0].keys == ["neon", "postgres"]

    all_lore = await store2.list_lorebooks()
    assert "db_lore" in all_lore

    await store2.delete_lorebook("db_lore")
    assert await store2.get_lorebook("db_lore") is None


