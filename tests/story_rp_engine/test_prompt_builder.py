import pytest
from story_rp_engine.core.types import CharacterCard, LorebookEntry
from story_rp_engine.rp.prompt_builder import (
    build_rp_system_instruction,
    replace_macros,
)

def test_build_rp_system_instruction_macro_replacement():
    card = CharacterCard(
        char_id="aria",
        name="Aria",
        description="{{char}} is an elven scout.",
        personality="Brave",
        scenario="{{user}} meets {{char}} in the woods.",
        first_mes="Halt!",
        mes_example="<START>\n{{user}}: Hello\n{{char}}: Who are you?",
    )
    lore = [LorebookEntry(keys=["woods"], content="The Whispering Woods are ancient.")]

    instruction = build_rp_system_instruction(card, active_lore=lore, user_name="Alex")
    assert "Aria is an elven scout." in instruction
    assert "Alex meets Aria in the woods." in instruction
    assert "The Whispering Woods are ancient." in instruction
    assert "{{char}}" not in instruction
    assert "{{user}}" not in instruction


def test_build_rp_system_instruction_with_post_history_instructions():
    card = CharacterCard(
        char_id="seraphina",
        name="Seraphina",
        description="{{char}} is a scholar.",
        personality="Calm",
        scenario="A library archive.",
        first_mes="Welcome.",
        mes_example="",
        post_history_instructions="Maintain a polite tone with {{user}}.",
    )

    instruction = build_rp_system_instruction(card, active_lore=None, user_name="Morgan")
    assert "### Additional Directives\nMaintain a polite tone with Morgan." in instruction
    assert "### World Information & Lore" not in instruction
    assert "### Dialogue Examples" not in instruction


def test_build_rp_system_instruction_default_user_name():
    card = CharacterCard(
        char_id="bob",
        name="Bob",
        description="Builder",
        personality="Helpful",
        scenario="Building a house.",
        first_mes="Can we fix it?",
        mes_example="",
    )

    instruction = build_rp_system_instruction(card)
    assert "speak for User." in instruction


def test_replace_macros_edge_cases():
    assert replace_macros("", "Aria", "Alex") == ""
    assert replace_macros(None, "Aria", "Alex") == ""
    text = "{{char}} smiled at {{user}}. 'Greetings, {{user}}!' said {{char}}."
    expected = "Aria smiled at Alex. 'Greetings, Alex!' said Aria."
    assert replace_macros(text, "Aria", "Alex") == expected


def test_replace_macros_case_insensitive():
    text = "{{Char}} greeted {{USER}}. {{CHAR}} nodded to {{User}}."
    expected = "Aria greeted Alex. Aria nodded to Alex."
    result = replace_macros(text, "Aria", "Alex")
    assert result == expected
    assert "{{char}}" not in result.lower()
    assert "{{user}}" not in result.lower()




