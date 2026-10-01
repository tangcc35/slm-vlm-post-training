import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Dict, Optional

from google.adk.sessions import BaseSessionService, DatabaseSessionService, Session
from sqlalchemy import Column, DateTime, String, Text, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import declarative_base

from story_rp_engine.core.types import CharacterCard, Lorebook

logger = logging.getLogger("story_rp_engine.storage")

StoreBase = declarative_base()


class CharacterRecord(StoreBase):
    __tablename__ = "story_rp_characters"
    char_id = Column(String(128), primary_key=True)
    card_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class LorebookRecord(StoreBase):
    __tablename__ = "story_rp_lorebooks"
    lorebook_id = Column(String(128), primary_key=True)
    lorebook_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def _sanitize_key(key: str) -> str:
    cleaned = key.strip()
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("Invalid ID: path traversal characters not allowed")
    if not cleaned:
        raise ValueError("Invalid ID: empty key")
    return cleaned


class _AsyncWrapper:
    def __init__(self, coro_fn):
        self._coro_fn = coro_fn

    def __await__(self):
        return self._coro_fn().__await__()


class EngineStore:
    def __init__(
        self,
        storage_dir: str = ".engine_data",
        session_service: Optional[BaseSessionService] = None,
        db_url: Optional[str] = None,
    ):
        self.storage_dir = storage_dir
        self.char_dir = os.path.join(storage_dir, "characters")
        self.lorebooks_dir = os.path.join(storage_dir, "lorebooks")
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.char_dir, exist_ok=True)
        os.makedirs(self.lorebooks_dir, exist_ok=True)

        self._character_cache: Dict[str, CharacterCard] = {}
        self._lorebook_cache: Dict[str, Lorebook] = {}
        self._table_lock = asyncio.Lock()
        self._tables_prepared = False

        if session_service is not None:
            self.session_service = session_service
        else:
            resolved_db_url = db_url or f"sqlite+aiosqlite:///{os.path.abspath(os.path.join(storage_dir, 'sessions.db'))}"
            self.session_service = DatabaseSessionService(db_url=resolved_db_url)

        self.db_engine = (
            getattr(self.session_service, "db_engine", None)
            if isinstance(self.session_service, DatabaseSessionService)
            else None
        )
        if self.db_engine is not None:
            self.session_factory = async_sessionmaker(
                bind=self.db_engine, expire_on_commit=False
            )
        else:
            self.session_factory = None

    async def prepare_tables(self) -> None:
        if self._tables_prepared or self.db_engine is None:
            return
        async with self._table_lock:
            if self._tables_prepared:
                return
            async with self.db_engine.begin() as conn:
                await conn.run_sync(StoreBase.metadata.create_all)
            self._tables_prepared = True

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

    # Character Card persistence methods (sync & async)

    def save_character_sync(self, char_id: str, card: CharacterCard) -> None:
        char_id = _sanitize_key(char_id)
        path = os.path.join(self.char_dir, f"{char_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(card.model_dump_json(indent=2))
        self._character_cache[char_id] = card

    def save_character(self, char_id: str, card: CharacterCard):
        char_id = _sanitize_key(char_id)
        self.save_character_sync(char_id, card)

        async def _async_db():
            if self.db_engine is not None and self.session_factory is not None:
                await self.prepare_tables()
                json_str = card.model_dump_json(indent=2)
                async with self.session_factory() as session:
                    record = await session.get(CharacterRecord, char_id)
                    if record is not None:
                        record.card_json = json_str
                        record.updated_at = datetime.now(timezone.utc)
                    else:
                        record = CharacterRecord(char_id=char_id, card_json=json_str)
                        session.add(record)
                    await session.commit()

        return _AsyncWrapper(_async_db)

    def get_character_sync(self, char_id: str) -> Optional[CharacterCard]:
        char_id = _sanitize_key(char_id)
        if char_id in self._character_cache:
            return self._character_cache[char_id]
        path = os.path.join(self.char_dir, f"{char_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        card = CharacterCard.model_validate(data)
        self._character_cache[char_id] = card
        return card

    async def get_character(self, char_id: str) -> Optional[CharacterCard]:
        char_id = _sanitize_key(char_id)
        if char_id in self._character_cache:
            return self._character_cache[char_id]

        if self.db_engine is not None and self.session_factory is not None:
            await self.prepare_tables()
            async with self.session_factory() as session:
                record = await session.get(CharacterRecord, char_id)
                if record is not None:
                    card = CharacterCard.model_validate_json(record.card_json)
                    self._character_cache[char_id] = card
                    return card
                # Not found in authoritative database; remove any stale local file
                path = os.path.join(self.char_dir, f"{char_id}.json")
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return None

        return self.get_character_sync(char_id)

    def list_characters_sync(self) -> Dict[str, CharacterCard]:
        results = {}
        for filename in os.listdir(self.char_dir):
            if filename.endswith(".json"):
                char_id = filename[:-5]
                card = self.get_character_sync(char_id)
                if card:
                    results[char_id] = card
        for k, v in self._character_cache.items():
            if k not in results:
                results[k] = v
        return results

    async def list_characters(self) -> Dict[str, CharacterCard]:
        if self.db_engine is not None and self.session_factory is not None:
            await self.prepare_tables()
            results = {}
            async with self.session_factory() as session:
                stmt = select(CharacterRecord)
                rows = await session.execute(stmt)
                for record in rows.scalars():
                    try:
                        card = CharacterCard.model_validate_json(record.card_json)
                        results[record.char_id] = card
                        self._character_cache[record.char_id] = card
                    except Exception as e:
                        logger.warning("Failed to parse character %s from db: %s", record.char_id, e)
            return results

        return self.list_characters_sync()

    def delete_character_sync(self, char_id: str) -> None:
        char_id = _sanitize_key(char_id)
        self._character_cache.pop(char_id, None)
        path = os.path.join(self.char_dir, f"{char_id}.json")
        if os.path.exists(path):
            os.remove(path)

    async def delete_character(self, char_id: str) -> None:
        char_id = _sanitize_key(char_id)
        self._character_cache.pop(char_id, None)
        found_in_db = False
        if self.db_engine is not None and self.session_factory is not None:
            await self.prepare_tables()
            async with self.session_factory() as session:
                record = await session.get(CharacterRecord, char_id)
                if record is not None:
                    await session.delete(record)
                    await session.commit()
                    found_in_db = True

        path = os.path.join(self.char_dir, f"{char_id}.json")
        found_on_disk = os.path.exists(path)
        if found_on_disk:
            os.remove(path)

        if not found_in_db and not found_on_disk:
            raise KeyError(f"Character {char_id} not found")

    # Lorebook persistence methods (sync & async)

    def save_lorebook_sync(self, lorebook_id: str, lorebook: Lorebook) -> None:
        lorebook_id = _sanitize_key(lorebook_id)
        path = os.path.join(self.lorebooks_dir, f"{lorebook_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(lorebook.model_dump_json(indent=2))
        self._lorebook_cache[lorebook_id] = lorebook

    def save_lorebook(self, lorebook_id: str, lorebook: Lorebook):
        lorebook_id = _sanitize_key(lorebook_id)
        self.save_lorebook_sync(lorebook_id, lorebook)

        async def _async_db():
            if self.db_engine is not None and self.session_factory is not None:
                await self.prepare_tables()
                json_str = lorebook.model_dump_json(indent=2)
                async with self.session_factory() as session:
                    record = await session.get(LorebookRecord, lorebook_id)
                    if record is not None:
                        record.lorebook_json = json_str
                        record.updated_at = datetime.now(timezone.utc)
                    else:
                        record = LorebookRecord(lorebook_id=lorebook_id, lorebook_json=json_str)
                        session.add(record)
                    await session.commit()

        return _AsyncWrapper(_async_db)

    def get_lorebook_sync(self, lorebook_id: str) -> Optional[Lorebook]:
        lorebook_id = _sanitize_key(lorebook_id)
        if lorebook_id in self._lorebook_cache:
            return self._lorebook_cache[lorebook_id]
        path = os.path.join(self.lorebooks_dir, f"{lorebook_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lb = Lorebook.model_validate(data)
        self._lorebook_cache[lorebook_id] = lb
        return lb

    async def get_lorebook(self, lorebook_id: str) -> Optional[Lorebook]:
        lorebook_id = _sanitize_key(lorebook_id)
        if lorebook_id in self._lorebook_cache:
            return self._lorebook_cache[lorebook_id]

        if self.db_engine is not None and self.session_factory is not None:
            await self.prepare_tables()
            async with self.session_factory() as session:
                record = await session.get(LorebookRecord, lorebook_id)
                if record is not None:
                    lb = Lorebook.model_validate_json(record.lorebook_json)
                    self._lorebook_cache[lorebook_id] = lb
                    return lb
                # Not found in authoritative database; remove any stale local file
                path = os.path.join(self.lorebooks_dir, f"{lorebook_id}.json")
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return None

        return self.get_lorebook_sync(lorebook_id)

    def list_lorebooks_sync(self) -> Dict[str, Lorebook]:
        results = {}
        for filename in os.listdir(self.lorebooks_dir):
            if filename.endswith(".json"):
                lb_id = filename[:-5]
                lb = self.get_lorebook_sync(lb_id)
                if lb:
                    results[lb_id] = lb
        for k, v in self._lorebook_cache.items():
            if k not in results:
                results[k] = v
        return results

    async def list_lorebooks(self) -> Dict[str, Lorebook]:
        if self.db_engine is not None and self.session_factory is not None:
            await self.prepare_tables()
            results = {}
            async with self.session_factory() as session:
                stmt = select(LorebookRecord)
                rows = await session.execute(stmt)
                for record in rows.scalars():
                    try:
                        lb = Lorebook.model_validate_json(record.lorebook_json)
                        results[record.lorebook_id] = lb
                        self._lorebook_cache[record.lorebook_id] = lb
                    except Exception as e:
                        logger.warning("Failed to parse lorebook %s from db: %s", record.lorebook_id, e)
            return results

        return self.list_lorebooks_sync()

    def delete_lorebook_sync(self, lorebook_id: str) -> None:
        lorebook_id = _sanitize_key(lorebook_id)
        self._lorebook_cache.pop(lorebook_id, None)
        path = os.path.join(self.lorebooks_dir, f"{lorebook_id}.json")
        if os.path.exists(path):
            os.remove(path)

    async def delete_lorebook(self, lorebook_id: str) -> None:
        lorebook_id = _sanitize_key(lorebook_id)
        self._lorebook_cache.pop(lorebook_id, None)
        found_in_db = False
        if self.db_engine is not None and self.session_factory is not None:
            await self.prepare_tables()
            async with self.session_factory() as session:
                record = await session.get(LorebookRecord, lorebook_id)
                if record is not None:
                    await session.delete(record)
                    await session.commit()
                    found_in_db = True

        path = os.path.join(self.lorebooks_dir, f"{lorebook_id}.json")
        found_on_disk = os.path.exists(path)
        if found_on_disk:
            os.remove(path)

        if not found_in_db and not found_on_disk:
            raise KeyError(f"Lorebook {lorebook_id} not found")
