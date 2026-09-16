import pytest
from story_rp_engine.core.types import CharacterCardV2, CharacterCardV2Data, ChatMessage, LorebookEntry
from story_rp_engine.rp.prompt_builder import (
    assemble_history_with_steering,
    build_rp_system_instruction,
    replace_macros,
)


def test_build_rp_system_instruction_macro_replacement():
    data = CharacterCardV2Data(
        name="Aria",
        description="{{char}} is an elven scout.",
        personality="Brave",
        scenario="{{user}} meets {{char}} in the woods.",
        first_mes="Halt!",
        mes_example="<START>\n{{user}}: Hello\n{{char}}: Who are you?",
    )
    card = CharacterCardV2(data=data)
    lore = [LorebookEntry(keys=["woods"], content="The Whispering Woods are ancient.")]

    instruction = build_rp_system_instruction(card, active_lore=lore, user_name="Alex")
    assert "Aria is an elven scout." in instruction
    assert "Alex meets Aria in the woods." in instruction
    assert "The Whispering Woods are ancient." in instruction
    assert "{{char}}" not in instruction
    assert "{{user}}" not in instruction


def test_build_rp_system_instruction_with_post_history_instructions():
    data = CharacterCardV2Data(
        name="Seraphina",
        description="{{char}} is a scholar.",
        personality="Calm",
        scenario="A library archive.",
        first_mes="Welcome.",
        mes_example="",
        post_history_instructions="Maintain a polite tone with {{user}}.",
    )
    card = CharacterCardV2(data=data)

    instruction = build_rp_system_instruction(card, active_lore=None, user_name="Morgan")
    assert "### Additional Directives\nMaintain a polite tone with Morgan." in instruction
    assert "### World Information & Lore" not in instruction
    assert "### Dialogue Examples" not in instruction


def test_build_rp_system_instruction_default_user_name():
    data = CharacterCardV2Data(
        name="Bob",
        description="Builder",
        personality="Helpful",
        scenario="Building a house.",
        first_mes="Can we fix it?",
        mes_example="",
    )
    card = CharacterCardV2(data=data)

    instruction = build_rp_system_instruction(card)
    assert "speak for User." in instruction


def test_replace_macros_edge_cases():
    assert replace_macros("", "Aria", "Alex") == ""
    assert replace_macros(None, "Aria", "Alex") == ""
    text = "{{char}} smiled at {{user}}. 'Greetings, {{user}}!' said {{char}}."
    expected = "Aria smiled at Alex. 'Greetings, Alex!' said Aria."
    assert replace_macros(text, "Aria", "Alex") == expected


def test_assemble_history_with_authors_note_injection():
    messages = [
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Turn 2"),
        ChatMessage(role="user", content="Turn 3"),
        ChatMessage(role="assistant", content="Turn 4"),
    ]
    # Depth = 2 means inserted 2 messages before the end
    assembled = assemble_history_with_steering(
        messages=messages,
        authors_note="[Style: poetic and slow]",
        depth=2,
        max_turns=10,
    )
    assert len(assembled) == 5
    # The note should be injected at index len(assembled) - 1 - 2 = 2
    assert assembled[2]["content"] == "[Style: poetic and slow]"
    assert assembled[2]["role"] == "system"
    assert assembled[3]["content"] == "Turn 3"
    assert assembled[4]["content"] == "Turn 4"


def test_assemble_history_without_authors_note():
    messages = [
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Turn 2"),
    ]
    # None note
    assembled_none = assemble_history_with_steering(messages=messages, authors_note=None)
    assert len(assembled_none) == 2
    assert assembled_none == [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Turn 2"},
    ]

    # Empty string or whitespace note
    assembled_empty = assemble_history_with_steering(messages=messages, authors_note="   ")
    assert len(assembled_empty) == 2


def test_assemble_history_truncation_max_turns():
    messages = [
        ChatMessage(role="user", content=f"Turn {i}")
        for i in range(1, 11)
    ]
    assembled = assemble_history_with_steering(messages=messages, authors_note=None, max_turns=3)
    assert len(assembled) == 3
    assert assembled[0]["content"] == "Turn 8"
    assert assembled[1]["content"] == "Turn 9"
    assert assembled[2]["content"] == "Turn 10"


def test_assemble_history_depth_boundary_conditions():
    messages = [
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Turn 2"),
    ]

    # depth <= 0 appends to end
    assembled_zero = assemble_history_with_steering(
        messages=messages,
        authors_note="[Steering at 0]",
        depth=0,
    )
    assert len(assembled_zero) == 3
    assert assembled_zero[-1]["content"] == "[Steering at 0]"
    assert assembled_zero[-1]["role"] == "system"

    # depth >= len(history) appends to end
    assembled_overflow = assemble_history_with_steering(
        messages=messages,
        authors_note="[Steering overflow]",
        depth=10,
    )
    assert len(assembled_overflow) == 3
    assert assembled_overflow[-1]["content"] == "[Steering overflow]"

    # Empty messages list with authors_note
    assembled_empty_history = assemble_history_with_steering(
        messages=[],
        authors_note="[Steering solo]",
        depth=2,
    )
    assert len(assembled_empty_history) == 1
    assert assembled_empty_history[0]["content"] == "[Steering solo]"
