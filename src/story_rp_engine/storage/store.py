import json
import os
from typing import Any, Dict, Optional
from google.adk.sessions import BaseSessionService, DatabaseSessionService, Session
from story_rp_engine.core.types import CharacterCard, Lorebook


def _sanitize_key(key: str) -> str:
    cleaned = key.strip()
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("Invalid ID: path traversal characters not allowed")
    if not cleaned:
        raise ValueError("Invalid ID: empty key")
    return cleaned


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

        if session_service is not None:
            self.session_service = session_service
        else:
            resolved_db_url = db_url or f"sqlite+aiosqlite:///{os.path.abspath(os.path.join(storage_dir, 'sessions.db'))}"
            self.session_service = DatabaseSessionService(db_url=resolved_db_url)

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

    def save_character(self, char_id: str, card: CharacterCard) -> None:
        char_id = _sanitize_key(char_id)
        path = os.path.join(self.char_dir, f"{char_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(card.model_dump_json(indent=2))

    def get_character(self, char_id: str) -> Optional[CharacterCard]:
        char_id = _sanitize_key(char_id)
        path = os.path.join(self.char_dir, f"{char_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CharacterCard.model_validate(data)

    def list_characters(self) -> Dict[str, CharacterCard]:
        results = {}
        for filename in os.listdir(self.char_dir):
            if filename.endswith(".json"):
                char_id = filename[:-5]
                card = self.get_character(char_id)
                if card:
                    results[char_id] = card
        return results

    def save_lorebook(self, lorebook_id: str, lorebook: Lorebook) -> None:
        lorebook_id = _sanitize_key(lorebook_id)
        path = os.path.join(self.lorebooks_dir, f"{lorebook_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(lorebook.model_dump_json(indent=2))

    def get_lorebook(self, lorebook_id: str) -> Optional[Lorebook]:
        lorebook_id = _sanitize_key(lorebook_id)
        path = os.path.join(self.lorebooks_dir, f"{lorebook_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Lorebook.model_validate(data)

    def list_lorebooks(self) -> Dict[str, Lorebook]:
        results = {}
        for filename in os.listdir(self.lorebooks_dir):
            if filename.endswith(".json"):
                lb_id = filename[:-5]
                lb = self.get_lorebook(lb_id)
                if lb:
                    results[lb_id] = lb
        return results
