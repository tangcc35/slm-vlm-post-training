import json
from typing import Any, Dict
from pydantic import ValidationError
from story_rp_engine.core.types import CharacterCard

def load_character_from_dict(payload: Dict[str, Any]) -> CharacterCard:
    try:
        return CharacterCard.model_validate(payload)
    except ValidationError as e:
        raise ValueError(f"Invalid Character Card: {e}") from e

def load_character_from_json(json_str: str) -> CharacterCard:
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Malformed JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Invalid Character Card: expected JSON object")
    return load_character_from_dict(data)
