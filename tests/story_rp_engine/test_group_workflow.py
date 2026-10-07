import pytest
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, GroupCard, Lorebook, LorebookEntry
from story_rp_engine.group.workflow import create_group_workflow, group_speakers, pick_speakers

ALICE = CharacterCard(char_id="alice", name="Alice", description="A bard.")
BOB = CharacterCard(char_id="bob", name="Bob", description="A blacksmith.")
GROUP = GroupCard(group_id="tavern", name="Tavern", char_ids=["alice", "bob"])


def _texts(llm_request):
    return ["".join(p.text or "" for p in c.parts) for c in llm_request.contents]


async def _run_turns(messages, state_delta=None):
    """Runs group turns in one session and returns the authors of its events."""
    service = InMemorySessionService()
    runner = Runner(
        app=App(name="group_app", root_agent=create_group_workflow(GROUP, [ALICE, BOB], EngineConfig())),
        session_service=service,
        auto_create_session=True,
    )
    for message in messages:
        async for _ in runner.run_async(
            user_id="User",
            session_id="s1",
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=message)]),
            state_delta=state_delta,
        ):
            pass
    session = await service.get_session(app_name="group_app", user_id="User", session_id="s1")
    return [ev.author for ev in session.events]


def test_pick_speakers_drops_repeats_and_falls_back_to_group_order():
    assert pick_speakers({"speakers": ["bob", "alice", "bob"]}, ["alice", "bob"]) == ["bob", "alice"]
    assert pick_speakers({"speakers": []}, ["alice", "bob"]) == ["alice", "bob"]
    assert pick_speakers(None, ["alice", "bob"]) == ["alice", "bob"]


def test_group_speakers_maps_agent_names_to_char_ids():
    assert group_speakers(GROUP) == {"char_alice": "alice", "char_bob": "bob"}


@pytest.mark.anyio
async def test_characters_reply_in_selected_order(group_models):
    group_models.plan = {"speakers": ["bob", "alice"]}
    assert await _run_turns(["Hello, everyone."]) == ["user", "speaker_selector", "char_bob", "char_alice"]


@pytest.mark.anyio
async def test_characters_see_each_other_but_not_the_selector(group_models):
    group_models.plan = {"speakers": ["bob", "alice"]}
    await _run_turns(["Hello, everyone.", "How was the forge?"])

    assert _texts(group_models.requests["Alice"][0]) == ["Hello, everyone.", "For context:[char_bob] said: Bob line"]
    assert _texts(group_models.requests["Bob"][1]) == [
        "Hello, everyone.",
        "Bob line",
        "For context:[char_alice] said: Alice line",
        "How was the forge?",
    ]
    # The selector reads the characters' lines.
    assert "For context:[char_alice] said: Alice line" in _texts(group_models.requests["speaker_selector"][1])


@pytest.mark.anyio
async def test_selector_schema_allows_only_members(group_models):
    group_models.plan = {"speakers": ["alice"]}
    await _run_turns(["Hi."])

    schema = group_models.requests["speaker_selector"][0].config.response_schema
    assert schema.model_validate({"speakers": ["bob"]}).speakers == ["bob"]
    with pytest.raises(ValidationError):
        schema.model_validate({"speakers": ["zed"]})


@pytest.mark.anyio
async def test_characters_get_lore_and_authors_note(group_models):
    group_models.plan = {"speakers": ["alice"]}
    lorebook = Lorebook(name="World", entries=[LorebookEntry(keys=["forge"], content="The forge never cools.")])
    await _run_turns(["Is the forge lit?"], state_delta={"lorebook": lorebook.model_dump(), "authors_note": "Keep it short."})

    latest = _texts(group_models.requests["Alice"][0])[-1]
    assert latest.startswith("Is the forge lit?")
    assert "The forge never cools." in latest
    assert "Author's note: Keep it short." in latest
