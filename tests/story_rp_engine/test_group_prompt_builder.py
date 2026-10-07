from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.prompt_builder import (
    build_group_system_instruction,
    build_selector_instruction,
    group_agent_name,
    group_summary_prompt,
)

ALICE = CharacterCard(char_id="alice", name="Alice", description="Alice owes {{user}} a favor.", scenario="Alice's own scenario.")
BOB = CharacterCard(char_id="bob-2", name="Bob", description="A blacksmith.")
QI = CharacterCard(char_id="思琪", name="思琪", description="A poet.")
GROUP = GroupCard(group_id="tavern", name="Tavern", char_ids=["alice", "bob-2", "思琪"], scenario="{{user}} meets everyone at the tavern.")


def test_group_agent_name_keeps_ids_distinct():
    assert group_agent_name("alice") == "char_alice"
    assert group_agent_name("bob-2") == "char_bob_2"
    # Non-ASCII letters are valid in identifiers, so CJK IDs don't collapse to underscores.
    assert group_agent_name("思琪") == "char_思琪"
    assert group_agent_name("思琪") != group_agent_name("王五")
    assert group_agent_name("思琪").isidentifier()


def test_group_instruction_uses_group_scenario_and_introduces_the_others():
    text = build_group_system_instruction(ALICE, GROUP, [ALICE, BOB, QI], user_name="Sam", greeting="The fire crackles.")

    assert text.startswith("You are Alice in a group roleplay with Sam, Bob, 思琪.")
    assert "Alice owes Sam a favor." in text
    assert "Sam meets everyone at the tavern." in text
    assert "Alice's own scenario." not in text
    assert "- Bob (shown as [char_bob_2]): A blacksmith." in text
    assert "[char_alice]" not in text
    assert "The fire crackles." in text


def test_selector_instruction_lists_member_ids():
    text = build_selector_instruction(GROUP, [ALICE, BOB], user_name="Sam")

    assert "- alice: Alice (shown as [char_alice]). Alice owes Sam a favor." in text
    assert "- bob-2: Bob (shown as [char_bob_2]). A blacksmith." in text
    assert "Sam meets everyone at the tavern." in text
    assert '{"speakers"' in text


def test_group_summary_prompt_formats_with_braces_in_names():
    odd = CharacterCard(char_id="odd", name="{Odd}")
    filled = group_summary_prompt([ALICE, odd]).format(conversation_history="HISTORY")

    assert "- char_alice: Alice" in filled
    assert "- char_odd: {Odd}" in filled
    assert filled.endswith("HISTORY")
