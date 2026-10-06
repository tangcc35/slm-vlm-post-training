from story_rp_engine.core.types import Lorebook, LorebookEntry
from story_rp_engine.rp.lorebook import LorebookEngine

def test_lorebook_matching_keywords():
    entry1 = LorebookEntry(keys=["silver sword", "blade"], content="Forged with elven silver.", insertion_order=10)
    entry2 = LorebookEntry(keys=["dragon"], content="Ancient beast of myth.", insertion_order=20)
    lorebook = Lorebook(name="Mythology", entries=[entry1, entry2])

    # Text contains "silver sword"
    matched = LorebookEngine.find_matching_entries([lorebook], "He drew his silver sword quietly.")
    assert len(matched) == 1
    assert matched[0].content == "Forged with elven silver."

def test_lorebook_case_insensitivity():
    entry = LorebookEntry(keys=["Necromancer"], content="Practitioner of dark magic.")
    lorebook = Lorebook(name="Magic", entries=[entry])

    matched = LorebookEngine.find_matching_entries([lorebook], "The necromancer stood tall.")
    assert len(matched) == 1
    assert matched[0].content == "Practitioner of dark magic."

def test_lorebook_sorting_by_insertion_order():
    entry1 = LorebookEntry(keys=["sword"], content="Sword info", insertion_order=50)
    entry2 = LorebookEntry(keys=["shield"], content="Shield info", insertion_order=10)
    entry3 = LorebookEntry(keys=["armor"], content="Armor info", insertion_order=25)
    lorebook = Lorebook(name="Equipment", entries=[entry1, entry2, entry3])

    matched = LorebookEngine.find_matching_entries([lorebook], "He equipped his sword, shield, and armor.")
    assert len(matched) == 3
    assert [e.insertion_order for e in matched] == [10, 25, 50]
    assert [e.content for e in matched] == ["Shield info", "Armor info", "Sword info"]

def test_lorebook_disabled_entries_ignored():
    entry1 = LorebookEntry(keys=["potion"], content="Healing potion", enabled=False)
    entry2 = LorebookEntry(keys=["elixir"], content="Mana elixir", enabled=True)
    lorebook = Lorebook(name="Alchemy", entries=[entry1, entry2])

    matched = LorebookEngine.find_matching_entries([lorebook], "He drank the potion and elixir.")
    assert len(matched) == 1
    assert matched[0].content == "Mana elixir"

def test_lorebook_duplicate_content_deduped():
    entry1 = LorebookEntry(keys=["staff"], content="Wooden staff", insertion_order=10)
    entry2 = LorebookEntry(keys=["cane"], content="Wooden staff", insertion_order=20)
    lorebook = Lorebook(name="Items", entries=[entry1, entry2])

    matched = LorebookEngine.find_matching_entries([lorebook], "A staff and a cane were on the floor.")
    assert len(matched) == 1
    assert matched[0].content == "Wooden staff"

def test_lorebook_word_boundaries():
    entry = LorebookEntry(keys=["cat"], content="Feline creature")
    lorebook = Lorebook(name="Fauna", entries=[entry])

    # "cat" should not match "scatter" or "caterpillar"
    assert len(LorebookEngine.find_matching_entries([lorebook], "The caterpillar wandered into the scattered leaves.")) == 0
    # "cat" should match "The cat slept."
    assert len(LorebookEngine.find_matching_entries([lorebook], "The cat slept.")) == 1

def test_lorebook_multiple_books():
    book1 = Lorebook(name="Geography", entries=[LorebookEntry(keys=["Valoria"], content="Capital city", insertion_order=5)])
    book2 = Lorebook(name="History", entries=[LorebookEntry(keys=["Great War"], content="Conflict in 1042", insertion_order=2)])

    matched = LorebookEngine.find_matching_entries([book1, book2], "During the Great War, Valoria fell.")
    assert len(matched) == 2
    assert matched[0].content == "Conflict in 1042"
    assert matched[1].content == "Capital city"

def test_lorebook_no_match():
    lorebook = Lorebook(name="Empty", entries=[LorebookEntry(keys=["secret"], content="Hidden knowledge")])
    matched = LorebookEngine.find_matching_entries([lorebook], "Nothing of interest here.")
    assert matched == []

def test_lorebook_empty_inputs():
    assert LorebookEngine.find_matching_entries([], "Some text") == []
    lorebook = Lorebook(name="Test", entries=[LorebookEntry(keys=["test"], content="Test content")])
    assert LorebookEngine.find_matching_entries([lorebook], "") == []

def test_lorebook_empty_or_whitespace_keys():
    lorebook = Lorebook(name="Test", entries=[LorebookEntry(keys=["", "   "], content="Whitespace content")])
    assert LorebookEngine.find_matching_entries([lorebook], "Any regular text with spaces") == []


def test_lorebook_chinese_keys_match_inside_unspaced_text():
    lorebook = Lorebook(name="Mixed", entries=[
        LorebookEntry(keys=["长安"], content="Capital", insertion_order=10),
        LorebookEntry(keys=["KX-9"], content="Arm", insertion_order=20),
        LorebookEntry(keys=["bell"], content="Bell"),
    ])
    matched = LorebookEngine.find_matching_entries([lorebook], "我想去长安城，带着KX-9义体，听bells响")
    assert [e.content for e in matched] == ["Capital", "Arm"]
