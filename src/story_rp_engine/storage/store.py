import json
import os
from typing import Dict, List, Optional
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, Lorebook


def _sanitize_key(key: str) -> str:
    cleaned = key.strip()
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("Invalid ID: path traversal characters not allowed")
    if not cleaned:
        raise ValueError("Invalid ID: empty key")
    return cleaned


class EngineStore:
    def __init__(self, storage_dir: str = ".engine_data"):
        self.storage_dir = storage_dir
        self.char_dir = os.path.join(storage_dir, "characters")
        self.lorebooks_dir = os.path.join(storage_dir, "lorebooks")
        self.sessions_dir = os.path.join(storage_dir, "sessions")
        os.makedirs(self.char_dir, exist_ok=True)
        os.makedirs(self.lorebooks_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)

    def save_character(self, char_id: str, card: CharacterCardV2) -> None:
        char_id = _sanitize_key(char_id)
        path = os.path.join(self.char_dir, f"{char_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(card.model_dump_json(indent=2))

    def get_character(self, char_id: str) -> Optional[CharacterCardV2]:
        char_id = _sanitize_key(char_id)
        path = os.path.join(self.char_dir, f"{char_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CharacterCardV2.model_validate(data)

    def list_characters(self) -> Dict[str, CharacterCardV2]:
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

    def save_history(self, session_id: str, messages: List[ChatMessage]) -> None:
        session_id = _sanitize_key(session_id)
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        data = [msg.model_dump(mode="json") for msg in messages]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_history(self, session_id: str) -> List[ChatMessage]:
        session_id = _sanitize_key(session_id)
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChatMessage.model_validate(item) for item in data]
