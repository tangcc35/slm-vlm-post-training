from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest
from story_rp_engine.core import agent_registry
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.agent_utils import execute_runner_turn
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import Lorebook, LorebookEntry
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.story import director_agent, writer_agent
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent
from story_rp_engine.story.workflow import create_story_workflow


@pytest.fixture
def fake_models(monkeypatch):
    """Gives the director, writer and compaction summarizer fake models that reply '<name> output'.

    Returns {name: [llm_request, ...]}.
    """
    requests = {}

    class RecordingLlm(BaseLlm):
        model: str = "mock"
        agent_name: str = ""

        async def generate_content_async(self, llm_request, stream=False):
            requests.setdefault(self.agent_name, []).append(llm_request)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=f"{self.agent_name} output")]))

    monkeypatch.setattr(director_agent, "get_adk_model", lambda config: RecordingLlm(agent_name="story_director"))
    monkeypatch.setattr(writer_agent, "get_adk_model", lambda config: RecordingLlm(agent_name="story_writer"))
    monkeypatch.setattr(agent_registry, "get_adk_model", lambda config: RecordingLlm(agent_name="summarizer"))
    return requests


def _texts(llm_request):
    return ["".join(p.text or "" for p in c.parts) for c in llm_request.contents]


async def _run_turns(config, instructions, state_delta=None):
    runner = Runner(
        app=App(name="story_app", root_agent=create_story_workflow(config)),
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    for instruction in instructions:
        async for _ in runner.run_async(
            user_id="User",
            session_id="s1",
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=instruction)]),
            state_delta=state_delta or {"premise": "P", "genre": "G", "tone": "T", "current_text": "Rain.", "instruction": instruction},
        ):
            pass


def test_create_director_and_writer_agents():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    assert director.name == "story_director"
    assert "{premise?}" in director.instruction
    assert "{genre?}" in director.instruction
    assert "{tone?}" in director.instruction
    assert "{current_text?}" in director.instruction

    assert writer.name == "story_writer"
    assert "{premise?}" in writer.instruction
    assert "{genre?}" in writer.instruction
    assert "{tone?}" in writer.instruction
    assert "{current_text?}" in writer.instruction


def test_format_story_input_removed_from_workflow():
    import story_rp_engine.story.workflow as wf_mod
    assert not hasattr(wf_mod, "format_story_input")
    assert not hasattr(wf_mod, "format_story_director_input")


@pytest.mark.anyio
async def test_director_reads_history_and_writer_gets_only_notes(fake_models):
    await _run_turns(EngineConfig(), ["Introduce Mara.", "Mara finds the map."])

    assert _texts(fake_models["story_director"][1]) == [
        "Introduce Mara.",
        "story_director output",
        "For context:[story_writer] said: story_writer output",
        "Mara finds the map.",
    ]
    assert _texts(fake_models["story_writer"][1]) == ["story_director output"]


@pytest.mark.anyio
async def test_director_reads_compaction_summary(tmp_path, fake_models):
    registry = AgentRegistry(EngineConfig(compaction_interval=2), EngineStore(storage_dir=str(tmp_path)))
    runner = registry.get_story_runner()

    replies = [
        await execute_runner_turn(runner, "User", "s1", instruction, author="story_writer")
        for instruction in ["One.", "Two.", "Three."]
    ]

    assert replies == ["story_writer output"] * 3
    assert _texts(fake_models["summarizer"][0])[0].startswith("Below is part of a story-writing session")
    assert _texts(fake_models["story_director"][2]) == ["summarizer output", "Three."]


@pytest.mark.anyio
async def test_session_state_injection_into_workflow_instructions(fake_models):
    await _run_turns(
        EngineConfig(model_name="ollama/llama3.1:8b"),
        ["Wake the dragon."],
        state_delta={
            "premise": "A dragon sleeps in the cave.",
            "genre": "Fantasy",
            "tone": "Epic",
            "current_text": "The torch flickers in the damp air.",
            "instruction": "Wake the dragon.",
        },
    )

    director_inst = fake_models["story_director"][0].config.system_instruction
    assert "Premise: A dragon sleeps in the cave." in director_inst
    assert "Genre: Fantasy" in director_inst
    assert "Tone: Epic" in director_inst
    assert "The torch flickers in the damp air." in director_inst

    writer_inst = fake_models["story_writer"][0].config.system_instruction
    assert "A dragon sleeps in the cave." in writer_inst
    assert "Fantasy" in writer_inst
    assert "Epic" in writer_inst
    assert "The torch flickers in the damp air." in writer_inst


@pytest.mark.anyio
async def test_writer_prompt_includes_user_instruction(fake_models):
    await _run_turns(EngineConfig(), ["Mara confesses she stole the map."])
    assert "Mara confesses she stole the map." in fake_models["story_writer"][0].config.system_instruction


@pytest.mark.anyio
async def test_story_agents_use_configured_sampling(fake_models):
    await _run_turns(EngineConfig(temperature=0.3, top_p=0.5, max_tokens=700), ["Go on."])
    for name in ["story_director", "story_writer"]:
        cfg = fake_models[name][0].config
        assert (cfg.temperature, cfg.top_p, cfg.max_output_tokens) == (0.3, 0.5, 700)


@pytest.mark.anyio
async def test_director_gets_matching_lore_with_latest_message(fake_models):
    lorebook = Lorebook(
        name="World",
        entries=[
            LorebookEntry(keys=["Mara"], content="Mara is a one-eyed smuggler."),
            LorebookEntry(keys=["map"], content="The map is drawn on whale skin."),
            LorebookEntry(keys=["dragon"], content="Dragons sleep under Mount Ash."),
        ],
    )
    await _run_turns(
        EngineConfig(),
        ["She finds the map."],
        state_delta={"current_text": "Mara walked in.", "instruction": "She finds the map.", "lorebook": lorebook.model_dump()},
    )

    director = fake_models["story_director"][0]
    latest = _texts(director)[-1]
    # Keys are matched in the user's message only, not the recent text.
    assert latest.startswith("She finds the map.")
    assert "The map is drawn on whale skin." in latest
    assert "Mara is a one-eyed smuggler." not in latest
    assert "Dragons sleep under Mount Ash." not in latest
    assert "whale skin" not in director.config.system_instruction
    # The writer gets lore only through the director's notes.
    assert "whale skin" not in str(fake_models["story_writer"][0])
