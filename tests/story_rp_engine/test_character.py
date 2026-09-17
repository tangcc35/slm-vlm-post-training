import json
import pytest
from story_rp_engine.rp.character import load_character_from_dict, load_character_from_json

def test_load_character_from_valid_v2_dict():
    payload = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
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
    }
    card = load_character_from_dict(payload)
    assert card.spec == "chara_card_v2"
    assert card.spec_version == "2.0"
    assert card.data.name == "Kaelen"
    assert card.data.description == "Ranger of the North"
    assert card.data.system_prompt == "You are Kaelen."
    assert card.data.alternate_greetings == ["Greetings, traveler."]
    assert card.data.tags == ["fantasy", "ranger"]

def test_load_character_from_json_string():
    json_str = json.dumps({
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Elysia",
            "description": "Mage",
            "personality": "Curious",
            "scenario": "Tower",
            "first_mes": "Enter.",
            "mes_example": "",
        }
    })
    card = load_character_from_json(json_str)
    assert card.data.name == "Elysia"
    assert card.data.description == "Mage"
    assert card.data.personality == "Curious"
    assert card.data.scenario == "Tower"
    assert card.data.first_mes == "Enter."
    assert card.data.mes_example == ""

def test_load_character_missing_fields_raises():
    with pytest.raises(ValueError, match="Invalid Character Card V2"):
        load_character_from_dict({"spec": "chara_card_v2"})

def test_load_character_invalid_spec_raises():
    payload = {
        "spec": "chara_card_v1",
        "spec_version": "2.0",
        "data": {
            "name": "Elysia",
            "description": "Mage",
            "personality": "Curious",
            "scenario": "Tower",
            "first_mes": "Enter.",
            "mes_example": "",
        }
    }
    with pytest.raises(ValueError, match="Invalid Character Card V2"):
        load_character_from_dict(payload)

def test_load_character_malformed_json_raises():
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_character_from_json("{invalid_json: true,}")

def test_load_character_non_dict_json_raises():
    with pytest.raises(ValueError, match="expected JSON object"):
        load_character_from_json('"just a string"')

