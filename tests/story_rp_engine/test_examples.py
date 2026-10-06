import json
from pathlib import Path
from story_rp_engine.api.routes_lorebook import _slugify
from story_rp_engine.core.types import CharacterCard, Lorebook
from story_rp_engine.rp.lorebook import LorebookEngine

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "story_rp"


def _lorebooks() -> dict[str, Lorebook]:
    return {p.stem: Lorebook.model_validate_json(p.read_text()) for p in (EXAMPLES / "lorebooks").glob("*.json")}


def test_example_ids_match_file_names():
    # The demo runner finds cards and lorebooks by file name, and the API derives a lorebook's id from its name.
    for path in (EXAMPLES / "characters").glob("*.json"):
        assert CharacterCard.model_validate_json(path.read_text()).char_id == path.stem
    for lorebook_id, lorebook in _lorebooks().items():
        assert _slugify(lorebook.name) == lorebook_id


def test_demos_trigger_every_enabled_lore_entry():
    lorebooks = _lorebooks()
    triggered = set()
    for path in (EXAMPLES / "demos").glob("*.json"):
        demo = json.loads(path.read_text())
        if demo["mode"] == "rp":
            assert (EXAMPLES / "characters" / f"{demo['char_id']}.json").exists()
            lorebook_id, texts = demo["lorebook_id"], [t["message"] for t in demo["turns"]]
        else:
            lorebook_id, texts = demo["setup"]["lorebook_id"], [t["instruction"] for t in demo["turns"]]
        for text in texts:
            for entry in LorebookEngine.find_matching_entries([lorebooks[lorebook_id]], text):
                triggered.add((lorebook_id, entry.content))

    enabled = {(lb_id, e.content) for lb_id, lb in lorebooks.items() for e in lb.entries if e.enabled}
    assert triggered == enabled
