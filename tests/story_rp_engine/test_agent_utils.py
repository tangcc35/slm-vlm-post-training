import pytest
from story_rp_engine.core.agent_utils import format_sse_stream


@pytest.mark.anyio
async def test_format_sse_stream_instant_single_token():
    async def mock_gen():
        for token in ["Hello", " ", "world", "!"]:
            yield token

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=1)]
    assert len(events) == 6  # 4 deltas + 1 full_text + 1 [DONE]
    assert '{"delta": "Hello"}' in events[0]
    assert '{"delta": " "}' in events[1]
    assert '{"delta": "world"}' in events[2]
    assert '{"delta": "!"}' in events[3]
    assert '{"full_text": "Hello world!", "done": true}' in events[4]
    assert events[5] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_format_sse_stream_gemini_multiword_instant():
    async def mock_gen():
        chunks = [
            "The old clocktower groaned in the wind.",
            " Rain pelted the cold stone steps.",
        ]
        for c in chunks:
            yield c

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=1)]
    assert len(events) == 4  # 2 deltas + 1 full_text + 1 [DONE]
    assert '{"delta": "The old clocktower groaned in the wind."}' in events[0]
    assert '{"delta": " Rain pelted the cold stone steps."}' in events[1]
    assert '{"full_text": "The old clocktower groaned in the wind. Rain pelted the cold stone steps.", "done": true}' in events[2]
    assert events[3] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_format_sse_stream_token_threshold_flushes_before_stream_end():
    async def mock_gen():
        # Each chunk has ~6 words
        yield "The ancient scholar opened the tome."  # 6 words (accum: 6)
        yield " Dust flew from the yellowed pages."  # 6 words (accum: 12)
        yield " Arcane runes began to softly glow."  # 6 words (accum: 18 -> exceeds 16 threshold)
        yield " Shadows danced on the stone wall."  # 6 words (accum: 6)

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=16)]
    # Should yield:
    # 1. Delta for first 3 chunks combined (18 words >= 16 tokens)
    # 2. Delta for 4th chunk at stream end
    # 3. full_text
    # 4. [DONE]
    assert len(events) == 4
    assert 'The ancient scholar opened the tome. Dust flew from the yellowed pages. Arcane runes began to softly glow.' in events[0]
    assert 'Shadows danced on the stone wall.' in events[1]
    assert events[3] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_format_sse_stream_newline_immediate_flush():
    async def mock_gen():
        yield "First paragraph.\n"  # 2 words + newline -> immediate flush even with chunk_size=32
        yield "Second paragraph."

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=32)]
    assert len(events) == 4
    assert '{"delta": "First paragraph.\\n"}' in events[0]
    assert '{"delta": "Second paragraph."}' in events[1]
    assert events[3] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_format_sse_stream_ignores_empty_chunks():
    async def mock_gen():
        yield ""
        yield "Valid chunk."
        yield ""

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=1)]
    assert len(events) == 3
    assert '{"delta": "Valid chunk."}' in events[0]
    assert '{"full_text": "Valid chunk.", "done": true}' in events[1]
    assert events[2] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_format_sse_stream_reports_model_error():
    async def mock_gen():
        yield "Partial"
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=1)]
    assert events == [
        'data: {"delta": "Partial"}\n\n',
        'data: {"error": "429 RESOURCE_EXHAUSTED"}\n\n',
        "data: [DONE]\n\n",
    ]


@pytest.mark.anyio
async def test_format_sse_stream_labels_group_replies_by_speaker():
    async def mock_gen():
        yield ("bob", "Bob ")
        yield ("bob", "line")
        yield ("alice", "Alice line")

    events = [ev async for ev in format_sse_stream(mock_gen(), chunk_size=16)]
    assert events == [
        'data: {"speaker": "bob", "delta": "Bob line"}\n\n',
        'data: {"speaker": "alice", "delta": "Alice line"}\n\n',
        'data: {"replies": [{"speaker": "bob", "text": "Bob line"}, {"speaker": "alice", "text": "Alice line"}], "done": true}\n\n',
        "data: [DONE]\n\n",
    ]


@pytest.mark.anyio
async def test_stream_group_turn_yields_speaker_chunks_without_the_selector(group_models):
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from story_rp_engine.core.agent_utils import stream_group_turn
    from story_rp_engine.core.config import EngineConfig
    from story_rp_engine.core.types import CharacterCard, GroupCard
    from story_rp_engine.group.workflow import create_group_workflow, group_speakers

    group_models.plan = {"speakers": ["bob", "alice"]}
    group = GroupCard(group_id="g", name="G", char_ids=["alice", "bob"])
    cards = [CharacterCard(char_id="alice", name="Alice"), CharacterCard(char_id="bob", name="Bob")]
    runner = Runner(
        app=App(name="group_app", root_agent=create_group_workflow(group, cards, EngineConfig())),
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    chunks = [c async for c in stream_group_turn(runner, "User", "s1", "Hi.", speakers=group_speakers(group))]
    assert chunks == [("bob", "Bob "), ("bob", "line"), ("alice", "Alice "), ("alice", "line")]
