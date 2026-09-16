import json
from typing import Any, Dict
from pydantic import ValidationError
from story_rp_engine.core.types import CharacterCardV2

def load_character_from_dict(payload: Dict[str, Any]) -> CharacterCardV2:
    try:
        return CharacterCardV2.model_validate(payload)
    except ValidationError as e:
        raise ValueError(f"Invalid Character Card V2: {e}") from e

def load_character_from_json(json_str: str) -> CharacterCardV2:
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Malformed JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Invalid Character Card V2: expected JSON object")
    return load_character_from_dict(data)
