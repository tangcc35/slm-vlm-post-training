import json
import pytest
from story_rp_engine.rp.character import load_character_from_dict, load_character_from_json

def test_load_character_from_valid_dict():
    payload = {
        "char_id": "kaelen",
        "name": "Kaelen",
        "description": "Ranger of the North",
        "personality": "Stoic, loyal",
        "scenario": "A campfire in the woods",
        "first_mes": "Keep your voice down. Something is out there.",
        "mes_example": "{{user}}: What was that?\n{{char}}: Wolves, likely.",
        "system_prompt": "You are Kaelen.",
        "alternate_greetings": ["Greetings, traveler."],
        "tags": ["fantasy", "ranger"],
    }
    card = load_character_from_dict(payload)
    assert card.char_id == "kaelen"
    assert card.name == "Kaelen"
    assert card.description == "Ranger of the North"
    assert card.system_prompt == "You are Kaelen."
    assert card.alternate_greetings == ["Greetings, traveler."]
    assert card.tags == ["fantasy", "ranger"]

def test_load_character_from_json_string():
    json_str = json.dumps({
        "char_id": "elysia",
        "name": "Elysia",
        "description": "Mage",
        "personality": "Curious",
        "scenario": "Tower",
        "first_mes": "Enter.",
        "mes_example": "",
    })
    card = load_character_from_json(json_str)
    assert card.char_id == "elysia"
    assert card.name == "Elysia"
    assert card.description == "Mage"
    assert card.personality == "Curious"
    assert card.scenario == "Tower"
    assert card.first_mes == "Enter."
    assert card.mes_example == ""

def test_load_character_missing_fields_raises():
    with pytest.raises(ValueError, match="Invalid Character Card"):
        load_character_from_dict({"description": "incomplete"})

def test_load_character_malformed_json_raises():
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_character_from_json("{invalid_json: true,}")

def test_load_character_non_dict_json_raises():
    with pytest.raises(ValueError, match="expected JSON object"):
        load_character_from_json('"just a string"')

