"""Loads the sample characters and lorebooks in examples/story_rp into a running engine and plays the demos.

    uv run python scripts/story_rp_examples.py                    # save the characters and lorebooks
    uv run python scripts/story_rp_examples.py --demo all         # ...then play every demo
    uv run python scripts/story_rp_examples.py --demo rp_wren_lamp_room --base-url http://localhost:8000

Each demo deletes its session first, plays its turns through the non-streaming endpoints, then titles the session
so it shows up in the web UI's history. Before each turn it prints the lorebook entries the engine will inject.
"""
import argparse
import json
import urllib.request
from pathlib import Path

from story_rp_engine.core.types import Lorebook
from story_rp_engine.rp.lorebook import LorebookEngine

EXAMPLES = Path(__file__).resolve().parents[1] / ".examples" / "story_rp"


def load(kind: str, name: str) -> dict:
    return json.loads((EXAMPLES / kind / f"{name}.json").read_text())


def call(base_url: str, method: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        base_url + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def seed(base_url: str) -> None:
    for path in sorted((EXAMPLES / "characters").glob("*.json")):
        print(call(base_url, "POST", "/api/v1/characters", json.loads(path.read_text())))
    for path in sorted((EXAMPLES / "lorebooks").glob("*.json")):
        print(call(base_url, "POST", "/api/v1/lorebooks", json.loads(path.read_text())))


def print_turn(text: str, turn: dict, lorebook: Lorebook, authors_note: str | None = None) -> None:
    print(f"\n> {text}\n  # {turn['showcase']}")
    lore = LorebookEngine.find_matching_entries([lorebook], text)
    if lore:
        print("  [lore] " + ", ".join(entry.keys[0] for entry in lore))
    if authors_note:
        print(f"  [author's note] {authors_note}")


def play_rp(base_url: str, demo: dict, lorebook: Lorebook) -> None:
    session_id = demo["session_id"]
    card = load("characters", demo["char_id"])
    greeting = [card["first_mes"], *card["alternate_greetings"]][demo["greeting_index"]]
    call(base_url, "DELETE", f"/api/v1/rp/sessions/{session_id}")
    print(greeting)
    authors_note = None
    for i, turn in enumerate(demo["turns"]):
        # Like the web UI, the author's note and greeting are resent every turn; the lorebook only once.
        authors_note = turn.get("authors_note", authors_note)
        body = {
            "char_id": demo["char_id"],
            "session_id": session_id,
            "message": turn["message"],
            "user_name": demo["user_name"],
            "greeting": greeting,
            "authors_note": authors_note or None,
        }
        if i == 0:
            body["lorebook_id"] = demo["lorebook_id"]
        print_turn(turn["message"], turn, lorebook, authors_note)
        print("\n" + call(base_url, "POST", "/api/v1/rp/chat", body)["reply"])
    call(base_url, "PATCH", f"/api/v1/rp/sessions/{session_id}", {"title": demo["title"]})


def play_story(base_url: str, demo: dict, lorebook: Lorebook) -> None:
    session_id = demo["session_id"]
    call(base_url, "DELETE", f"/api/v1/story/sessions/{session_id}")
    for i, turn in enumerate(demo["turns"]):
        # The setup goes with the first turn only; the engine keeps it in session state.
        body = {"session_id": session_id, "instruction": turn["instruction"], **(demo["setup"] if i == 0 else {})}
        print_turn(turn["instruction"], turn, lorebook)
        print("\n" + call(base_url, "POST", "/api/v1/story/expand", body)["expansion"])
    call(base_url, "PATCH", f"/api/v1/story/sessions/{session_id}", {"title": demo["title"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--demo", help="A file name in examples/story_rp/demos without .json, or 'all'")
    args = parser.parse_args()

    seed(args.base_url)
    if not args.demo:
        return
    names = sorted(p.stem for p in (EXAMPLES / "demos").glob("*.json")) if args.demo == "all" else [args.demo]
    for name in names:
        demo = load("demos", name)
        lorebook_id = demo["lorebook_id"] if demo["mode"] == "rp" else demo["setup"]["lorebook_id"]
        lorebook = Lorebook.model_validate(load("lorebooks", lorebook_id))
        print(f"\n===== {demo['title']} ({name}) =====")
        (play_rp if demo["mode"] == "rp" else play_story)(args.base_url, demo, lorebook)


if __name__ == "__main__":
    main()
