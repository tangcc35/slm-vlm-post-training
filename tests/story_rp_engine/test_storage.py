import pytest
from story_rp_engine.core.types import (
    CharacterCardV2,
    CharacterCardV2Data,
    ChatMessage,
    Lorebook,
    LorebookEntry,
)
from story_rp_engine.storage.store import EngineStore


def test_character_crud(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    card = CharacterCardV2(
        data=CharacterCardV2Data(
            name="Valerie",
            description="Alchemist",
            personality="Witty",
            scenario="Shop",
            first_mes="Can I help you?",
            mes_example="",
        )
    )
    store.save_character("valerie_1", card)

    retrieved = store.get_character("valerie_1")
    assert retrieved is not None
    assert retrieved.data.name == "Valerie"

    all_chars = store.list_characters()
    assert "valerie_1" in all_chars
    assert all_chars["valerie_1"].data.name == "Valerie"


def test_character_not_found(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    assert store.get_character("nonexistent") is None


def test_session_history(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    history = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
    ]
    store.save_history("session_123", history)

    loaded = store.get_history("session_123")
    assert len(loaded) == 2
    assert loaded[0].content == "Hi"
    assert loaded[1].content == "Hello!"


def test_session_history_empty_when_missing(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    assert store.get_history("missing_session") == []


def test_lorebook_crud(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    lorebook = Lorebook(
        name="Arcane Lore",
        description="Lore regarding magic",
        entries=[
            LorebookEntry(
                keys=["elixir", "potion"],
                content="Alchemical drafts with restorative properties.",
                insertion_order=10,
            )
        ],
    )
    store.save_lorebook("arcane_lore", lorebook)

    retrieved = store.get_lorebook("arcane_lore")
    assert retrieved is not None
    assert retrieved.name == "Arcane Lore"
    assert len(retrieved.entries) == 1
    assert retrieved.entries[0].keys == ["elixir", "potion"]

    all_lore = store.list_lorebooks()
    assert "arcane_lore" in all_lore
    assert all_lore["arcane_lore"].name == "Arcane Lore"

    assert store.get_lorebook("nonexistent_lore") is None


def test_session_history_append(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    initial = [ChatMessage(role="user", content="Turn 1")]
    store.save_history("session_append", initial)

    loaded = store.get_history("session_append")
    assert len(loaded) == 1

    updated = loaded + [ChatMessage(role="assistant", content="Response 1")]
    store.save_history("session_append", updated)

    reloaded = store.get_history("session_append")
    assert len(reloaded) == 2
    assert reloaded[1].content == "Response 1"


def test_list_ignores_non_json_files(tmp_path):
    store = EngineStore(storage_dir=str(tmp_path))
    # Write dummy non-json files
    (tmp_path / "characters" / "notes.txt").write_text("not a json")
    (tmp_path / "lorebooks" / "README.md").write_text("some docs")

    assert store.list_characters() == {}
    assert store.list_lorebooks() == {}

