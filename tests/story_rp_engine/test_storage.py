import asyncio
import itertools
import os
import uuid
import pytest
from google.adk.sessions import DatabaseSessionService, Session
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from story_rp_engine.core.types import (
    CharacterCard,
    GroupCard,
    Lorebook,
    LorebookEntry,
)
from story_rp_engine.storage.db import normalize_db_url
from story_rp_engine.storage.store import EngineStore


def _card(char_id: str = "valerie", name: str = "Valerie") -> CharacterCard:
    return CharacterCard(
        char_id=char_id,
        name=name,
        description="Alchemist",
        personality="Witty",
        scenario="Shop",
        first_mes="Can I help you?",
        mes_example="",
    )


@pytest.fixture
def postgres_url():
    """A throwaway database on the server named by STORY_RP_TEST_PG_URL, dropped afterwards."""
    base_url = os.getenv("STORY_RP_TEST_PG_URL")
    if not base_url:
        pytest.skip("set STORY_RP_TEST_PG_URL to a Postgres server to run Postgres storage tests")
    admin_url = normalize_db_url(base_url)
    db_name = f"story_rp_test_{uuid.uuid4().hex[:12]}"

    async def run_admin(sql: str) -> None:
        engine = create_async_engine(admin_url, poolclass=NullPool, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as conn:
                await conn.execute(text(sql))
        finally:
            await engine.dispose()

    asyncio.run(run_admin(f'CREATE DATABASE "{db_name}"'))
    yield make_url(admin_url).set(database=db_name).render_as_string(hide_password=False)
    asyncio.run(run_admin(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))


@pytest.fixture(params=["files", "sqlite", "postgres"])
def make_store(request, tmp_path):
    """Factory for stores sharing one backend, like separate (serverless) app instances."""
    if request.param == "files":
        return lambda: EngineStore(storage_dir=str(tmp_path))
    if request.param == "sqlite":
        db_url = f"sqlite+aiosqlite:///{tmp_path / 'engine.db'}"
    else:
        db_url = request.getfixturevalue("postgres_url")
    instance = itertools.count()
    return lambda: EngineStore(
        storage_dir=str(tmp_path / f"instance_{next(instance)}"),
        db_url=db_url,
        db_null_pool=True,
    )


@pytest.fixture
def db_store_pair(request, tmp_path):
    """Two stores on one database with separate local dirs, i.e. two cold-started instances."""
    if request.param == "sqlite":
        db_url = f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}"
    else:
        db_url = request.getfixturevalue("postgres_url")
    return (
        EngineStore(storage_dir=str(tmp_path / "instance_1"), db_url=db_url, db_null_pool=True),
        EngineStore(storage_dir=str(tmp_path / "instance_2"), db_url=db_url, db_null_pool=True),
    )


@pytest.mark.anyio
async def test_character_crud(make_store):
    store = make_store()
    await store.save_character("valerie_1", _card("valerie_1"))

    retrieved = await store.get_character("valerie_1")
    assert retrieved is not None
    assert retrieved.name == "Valerie"

    all_chars = await store.list_characters()
    assert "valerie_1" in all_chars
    assert all_chars["valerie_1"].name == "Valerie"

    assert await store.delete_character("valerie_1") is True
    assert await store.get_character("valerie_1") is None
    assert await store.delete_character("valerie_1") is False


@pytest.mark.anyio
async def test_character_not_found(make_store):
    store = make_store()
    assert await store.get_character("nonexistent") is None


@pytest.mark.anyio
async def test_save_character_overwrites_existing(make_store):
    store = make_store()
    await store.save_character("valerie", _card(name="Valerie"))
    await store.save_character("valerie", _card(name="Valerie the Second"))

    assert (await store.get_character("valerie")).name == "Valerie the Second"
    assert list(await store.list_characters()) == ["valerie"]


@pytest.mark.anyio
async def test_unicode_character_id(make_store):
    store = make_store()
    await store.save_character("思琪", _card("思琪", name="王"))
    assert (await store.get_character("思琪")).name == "王"
    assert "思琪" in await store.list_characters()


@pytest.mark.anyio
async def test_lorebook_crud(make_store):
    store = make_store()
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

    assert await store.delete_lorebook("arcane_lore") is True
    assert await store.get_lorebook("arcane_lore") is None
    assert await store.delete_lorebook("arcane_lore") is False


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
async def test_path_traversal_defense(make_store, bad_key):
    store = make_store()
    lorebook = Lorebook(name="Arcane", description="Lore", entries=[])

    # Verify all CRUD methods reject path traversal payloads
    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.save_character(bad_key, _card())

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.get_character(bad_key)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.delete_character(bad_key)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.save_lorebook(bad_key, lorebook)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.get_lorebook(bad_key)

    with pytest.raises(ValueError, match="Invalid ID: path traversal characters not allowed"):
        await store.delete_lorebook(bad_key)


@pytest.mark.anyio
async def test_key_whitespace_stripping(make_store):
    store = make_store()
    await store.save_character("  valerie_padded  ", _card())
    retrieved = await store.get_character("valerie_padded")
    assert retrieved is not None
    assert retrieved.name == "Valerie"


@pytest.mark.anyio
@pytest.mark.parametrize("db_store_pair", ["sqlite", "postgres"], indirect=True)
async def test_database_shared_across_instances(db_store_pair, tmp_path):
    first, second = db_store_pair
    await first.save_character("cloud_valerie", _card("cloud_valerie", name="Valerie in the Cloud"))
    await first.save_lorebook("db_lore", Lorebook(name="DB Lore", entries=[LorebookEntry(keys=["neon"], content="Serverless Postgres.")]))

    # A fresh instance with an empty local dir sees the data...
    assert (await second.get_character("cloud_valerie")).name == "Valerie in the Cloud"
    assert "cloud_valerie" in await second.list_characters()
    assert (await second.get_lorebook("db_lore")).entries[0].keys == ["neon"]

    # ...and edits/deletes from one instance are visible to the other (no stale copies).
    await second.save_character("cloud_valerie", _card("cloud_valerie", name="Valerie v2"))
    assert (await first.get_character("cloud_valerie")).name == "Valerie v2"

    assert await second.delete_character("cloud_valerie") is True
    assert await second.delete_lorebook("db_lore") is True
    assert await first.get_character("cloud_valerie") is None
    assert await first.list_lorebooks() == {}

    # Nothing was written to the instances' local storage dirs.
    assert not (tmp_path / "instance_1" / "characters").exists()
    assert not (tmp_path / "instance_2" / "lorebooks").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("db_store_pair", ["sqlite", "postgres"], indirect=True)
async def test_database_sessions_shared_across_instances(db_store_pair):
    first, second = db_store_pair
    await first.get_or_create_session(
        app_name="rp_app", user_id="User", session_id="sess_shared", initial_state={"char_id": "lyra"}
    )
    session = await second.session_service.get_session(app_name="rp_app", user_id="User", session_id="sess_shared")
    assert session is not None
    assert session.state.get("char_id") == "lyra"


@pytest.mark.anyio
@pytest.mark.parametrize("db_store_pair", ["sqlite", "postgres"], indirect=True)
async def test_database_rejects_overlong_ids(db_store_pair):
    store, _ = db_store_pair
    with pytest.raises(ValueError, match="Invalid ID: longer than 128 characters"):
        await store.save_character("x" * 129, _card())
    await store.save_character("x" * 128, _card())
    assert await store.get_character("x" * 128) is not None


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
async def test_group_crud(make_store):
    store = make_store()
    group = GroupCard(group_id="tavern", name="Tavern", char_ids=["valerie", "思琪"], scenario="A rainy night.")
    await store.save_group("tavern", group)

    assert await store.get_group("tavern") == group
    assert await store.list_groups() == {"tavern": group}

    assert await store.delete_group("tavern") is True
    assert await store.get_group("tavern") is None
    assert await store.delete_group("tavern") is False
