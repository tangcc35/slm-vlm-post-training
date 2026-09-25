import pytest
from pydantic import ValidationError
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import (
    CharacterCard,
    Lorebook,
    LorebookEntry,
    ChatMessage,
    StoryRequest,
    RPChatRequest,
)

def test_engine_config_defaults():
    config = EngineConfig()
    assert config.model_name == "ollama/llama3.1:8b"
    assert config.temperature == 0.8
    assert config.max_tokens == 131072

def test_character_card_valid():
    card = CharacterCard(
        char_id="seraphina",
        name="Seraphina",
        description="A quiet archivist",
        personality="Cautious, observant",
        scenario="An ancient library after hours",
        first_mes="Welcome to the archives. Please keep your voice down.",
        mes_example="<START>\n{{user}}: What is this place?\n{{char}}: It is memory preserved.",
    )
    assert card.char_id == "seraphina"
    assert card.name == "Seraphina"
    assert card.description == "A quiet archivist"
    assert card.personality == "Cautious, observant"
    assert card.scenario == "An ancient library after hours"
    assert card.first_mes == "Welcome to the archives. Please keep your voice down."
    assert card.mes_example == "<START>\n{{user}}: What is this place?\n{{char}}: It is memory preserved."
    assert card.alternate_greetings == []
    assert card.tags == []

def test_character_card_missing_char_id_raises():
    with pytest.raises(ValidationError):
        CharacterCard.model_validate({"name": "Elena"})

def test_character_card_missing_name_raises():
    with pytest.raises(ValidationError):
        CharacterCard.model_validate({"char_id": "elena"})

def test_lorebook_and_entry():
    entry = LorebookEntry(keys=["archive", "library"], content="The Archives were founded in 1420.")
    lorebook = Lorebook(name="Setting Lore", entries=[entry])
    assert len(lorebook.entries) == 1
    assert "archive" in lorebook.entries[0].keys

def test_story_request_defaults():
    req = StoryRequest(session_id="story_sess_1", current_text="The wind howled.")
    assert req.session_id == "story_sess_1"
    assert req.current_text == "The wind howled."
    assert req.genre == "Fiction"
    assert req.tone == "Balanced"
    assert req.max_tokens == 512
    assert req.chunk_size == 16

def test_story_request_custom_chunk_size():
    req = StoryRequest(session_id="story_sess_1", current_text="The wind howled.", chunk_size=8)
    assert req.session_id == "story_sess_1"
    assert req.chunk_size == 8

def test_story_request_missing_session_id_raises():
    with pytest.raises(ValidationError):
        StoryRequest(current_text="The wind howled.")

def test_story_request_empty_session_id_raises():
    with pytest.raises(ValidationError):
        StoryRequest(session_id="", current_text="The wind howled.")

def test_rp_chat_request_defaults():
    req = RPChatRequest(char_id="elena", session_id="s1", message="Hello")
    assert req.char_id == "elena"
    assert req.session_id == "s1"
    assert req.message == "Hello"
    assert req.user_name == "User"
    assert req.chunk_size == 16
    assert req.authors_note is None
