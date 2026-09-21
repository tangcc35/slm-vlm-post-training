import pytest
from pydantic import ValidationError
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import (
    CharacterCardV2,
    CharacterCardV2Data,
    Lorebook,
    LorebookEntry,
    ChatMessage,
    StoryRequest,
)

def test_engine_config_defaults():
    config = EngineConfig()
    assert config.model_name == "ollama/llama3.1:8b"
    assert config.temperature == 0.8
    assert config.max_tokens == 131072

def test_character_card_v2_valid():
    data = CharacterCardV2Data(
        name="Seraphina",
        description="A quiet archivist",
        personality="Cautious, observant",
        scenario="An ancient library after hours",
        first_mes="Welcome to the archives. Please keep your voice down.",
        mes_example="<START>\n{{user}}: What is this place?\n{{char}}: It is memory preserved.",
    )
    card = CharacterCardV2(data=data)
    assert card.spec == "chara_card_v2"
    assert card.spec_version == "2.0"
    assert card.data.name == "Seraphina"

def test_character_card_v2_invalid_spec():
    data = CharacterCardV2Data(
        name="A", description="B", personality="C", scenario="D", first_mes="E", mes_example="F"
    )
    with pytest.raises(ValidationError):
        CharacterCardV2(spec="invalid_spec", data=data)

def test_lorebook_and_entry():
    entry = LorebookEntry(keys=["archive", "library"], content="The Archives were founded in 1420.")
    lorebook = Lorebook(name="Setting Lore", entries=[entry])
    assert len(lorebook.entries) == 1
    assert "archive" in lorebook.entries[0].keys

def test_story_request_defaults():
    req = StoryRequest(current_text="The wind howled.")
    assert req.current_text == "The wind howled."
    assert req.genre == "Fiction"
    assert req.tone == "Balanced"
    assert req.max_tokens == 512
    assert req.chunk_size == 4

def test_story_request_custom_chunk_size():
    req = StoryRequest(current_text="The wind howled.", chunk_size=8)
    assert req.chunk_size == 8
