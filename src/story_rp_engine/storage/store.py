import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Generic, Optional, Type, TypeVar
from google.adk.sessions import BaseSessionService, DatabaseSessionService, Session
from pydantic import BaseModel, ValidationError
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, delete, insert, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from story_rp_engine.core.types import CharacterCard, Lorebook
from story_rp_engine.storage.db import engine_kwargs_for, normalize_db_url, redact_db_url

logger = logging.getLogger("story_rp_engine.storage")

ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_DB_KEY_LENGTH = 128

_metadata = MetaData()

# updated_at holds naive UTC, matching how ADK stores its own Postgres timestamps
# (asyncpg rejects timezone-aware values for TIMESTAMP WITHOUT TIME ZONE).
characters_table = Table(
    "story_rp_characters",
    _metadata,
    Column("char_id", String(MAX_DB_KEY_LENGTH), primary_key=True),
    Column("card_json", Text, nullable=False),
    Column("updated_at", DateTime),
)

lorebooks_table = Table(
    "story_rp_lorebooks",
    _metadata,
    Column("lorebook_id", String(MAX_DB_KEY_LENGTH), primary_key=True),
    Column("lorebook_json", Text, nullable=False),
    Column("updated_at", DateTime),
)


def _sanitize_key(key: str) -> str:
    cleaned = key.strip()
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("Invalid ID: path traversal characters not allowed")
    if not cleaned:
        raise ValueError("Invalid ID: empty key")
    return cleaned


class _FileCollection(Generic[ModelT]):
    """Stores each document as <directory>/<key>.json."""

    def __init__(self, directory: str, model: Type[ModelT]):
        self.directory = directory
        self.model = model
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.json")

    async def put(self, key: str, doc: ModelT) -> None:
        with open(self._path(key), "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))

    async def get(self, key: str) -> Optional[ModelT]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.model.model_validate(data)

    async def list(self) -> Dict[str, ModelT]:
        results = {}
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                key = filename[:-5]
                doc = await self.get(key)
                if doc:
                    results[key] = doc
        return results

    async def delete(self, key: str) -> bool:
        path = self._path(key)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True


class _SqlTables:
    """Creates the character and lorebook tables on first use."""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self._ready = False
        self._lock = asyncio.Lock()

    async def ensure(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            try:
                async with self.engine.begin() as conn:
                    await conn.run_sync(_metadata.create_all)
            except (IntegrityError, ProgrammingError):
                # Another instance created the tables concurrently; checkfirst now sees them.
                async with self.engine.begin() as conn:
                    await conn.run_sync(_metadata.create_all)
            self._ready = True


class _SqlCollection(Generic[ModelT]):
    """Stores each document as a JSON text row keyed by its ID."""

    def __init__(self, tables: _SqlTables, table: Table, key_column: str, json_column: str, model: Type[ModelT]):
        self.tables = tables
        self.table = table
        self.key = table.c[key_column]
        self.json = table.c[json_column]
        self.model = model

    async def put(self, key: str, doc: ModelT) -> None:
        if len(key) > MAX_DB_KEY_LENGTH:
            raise ValueError(f"Invalid ID: longer than {MAX_DB_KEY_LENGTH} characters")
        values = {self.json.name: doc.model_dump_json(), "updated_at": datetime.now(timezone.utc).replace(tzinfo=None)}
        await self.tables.ensure()
        engine = self.tables.engine
        async with engine.begin() as conn:
            if engine.dialect.name in ("postgresql", "sqlite"):
                dialect_insert = postgresql.insert if engine.dialect.name == "postgresql" else sqlite.insert
                stmt = dialect_insert(self.table).values({self.key.name: key, **values})
                await conn.execute(stmt.on_conflict_do_update(index_elements=[self.key], set_=values))
            else:
                result = await conn.execute(update(self.table).where(self.key == key).values(values))
                if result.rowcount == 0:
                    await conn.execute(insert(self.table).values({self.key.name: key, **values}))

    async def get(self, key: str) -> Optional[ModelT]:
        await self.tables.ensure()
        async with self.tables.engine.connect() as conn:
            raw = (await conn.execute(select(self.json).where(self.key == key))).scalar_one_or_none()
        return None if raw is None else self.model.model_validate_json(raw)

    async def list(self) -> Dict[str, ModelT]:
        await self.tables.ensure()
        async with self.tables.engine.connect() as conn:
            rows = (await conn.execute(select(self.key, self.json).order_by(self.key))).all()
        results = {}
        for key, raw in rows:
            try:
                results[key] = self.model.model_validate_json(raw)
            except ValidationError as e:
                logger.warning("Skipping unreadable row %r in %s: %s", key, self.table.name, e)
        return results

    async def delete(self, key: str) -> bool:
        await self.tables.ensure()
        async with self.tables.engine.begin() as conn:
            result = await conn.execute(delete(self.table).where(self.key == key))
        return result.rowcount > 0


class EngineStore:
    """Persists ADK sessions, character cards and lorebooks.

    With a database URL (e.g. Neon Postgres) everything lives in that database, so
    stateless serverless instances share it. Without one, characters and lorebooks
    are JSON files under storage_dir and sessions go to a local SQLite file.
    """

    def __init__(
        self,
        storage_dir: str = ".engine_data",
        session_service: Optional[BaseSessionService] = None,
        db_url: Optional[str] = None,
        db_null_pool: bool = False,
    ):
        self.storage_dir = storage_dir
        self.char_dir = os.path.join(storage_dir, "characters")
        self.lorebooks_dir = os.path.join(storage_dir, "lorebooks")
        self.db_url = normalize_db_url(db_url)
        engine_kwargs = engine_kwargs_for(self.db_url, db_null_pool) if self.db_url else {}

        if self.db_url is None:
            os.makedirs(self.storage_dir, exist_ok=True)
        else:
            logger.info("Using database %s", redact_db_url(self.db_url))

        engine: Optional[AsyncEngine] = None
        if session_service is not None:
            self.session_service = session_service
            if self.db_url:
                engine = create_async_engine(self.db_url, **engine_kwargs)
        else:
            session_db_url = self.db_url or f"sqlite+aiosqlite:///{os.path.abspath(os.path.join(storage_dir, 'sessions.db'))}"
            self.session_service = DatabaseSessionService(db_url=session_db_url, **engine_kwargs)
            if self.db_url:
                # Share ADK's engine so sessions and records use one connection setup.
                engine = self.session_service.db_engine

        if engine is None:
            self._characters = _FileCollection(self.char_dir, CharacterCard)
            self._lorebooks = _FileCollection(self.lorebooks_dir, Lorebook)
        else:
            tables = _SqlTables(engine)
            self._characters = _SqlCollection(tables, characters_table, "char_id", "card_json", CharacterCard)
            self._lorebooks = _SqlCollection(tables, lorebooks_table, "lorebook_id", "lorebook_json", Lorebook)

    async def get_or_create_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Session:
        session_id = _sanitize_key(session_id)
        user_id = _sanitize_key(user_id)
        session = await self.session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        if session is None:
            session = await self.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {},
            )
        return session

    async def save_character(self, char_id: str, card: CharacterCard) -> None:
        await self._characters.put(_sanitize_key(char_id), card)

    async def get_character(self, char_id: str) -> Optional[CharacterCard]:
        return await self._characters.get(_sanitize_key(char_id))

    async def list_characters(self) -> Dict[str, CharacterCard]:
        return await self._characters.list()

    async def delete_character(self, char_id: str) -> bool:
        """Deletes a character card; returns False if it did not exist."""
        return await self._characters.delete(_sanitize_key(char_id))

    async def save_lorebook(self, lorebook_id: str, lorebook: Lorebook) -> None:
        await self._lorebooks.put(_sanitize_key(lorebook_id), lorebook)

    async def get_lorebook(self, lorebook_id: str) -> Optional[Lorebook]:
        return await self._lorebooks.get(_sanitize_key(lorebook_id))

    async def list_lorebooks(self) -> Dict[str, Lorebook]:
        return await self._lorebooks.list()

    async def delete_lorebook(self, lorebook_id: str) -> bool:
        """Deletes a lorebook; returns False if it did not exist."""
        return await self._lorebooks.delete(_sanitize_key(lorebook_id))
