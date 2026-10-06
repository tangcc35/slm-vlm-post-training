from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.adk.runners import Runner
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    format_sse_stream,
    stream_runner_turn,
)
from story_rp_engine.core.types import StoryRequest

router = APIRouter(prefix="/api/v1/story", tags=["Story Co-Pilot"])

# The agents see only the end of the story (the writer's earlier replies); the director covers earlier
# parts through the (compacted) session history.
RECENT_TEXT_CHARS = 8000


def _get_story_runner(request: Request) -> Runner:
    runner = getattr(request.app.state, "runner", None)
    if runner is not None:
        return runner
    registry = getattr(request.app.state, "agent_registry", None)
    if registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    runner = registry.get_story_runner()
    request.app.state.runner = runner
    return runner


async def _story_turn_inputs(req: StoryRequest, request: Request) -> tuple[Runner, str, dict]:
    """Returns the runner, the user message and the state delta for one story chat turn."""
    if not req.session_id or not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    runner = _get_story_runner(request)
    instruction = req.instruction or "Continue the story naturally from the current point."
    # The story is the writer's replies so far; the setup fields come with the first turn and stay in state.
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id="User", session_id=req.session_id
    )
    passages = [
        "".join(p.text for p in ev.content.parts if p.text)
        for ev in (session.events if session else [])
        if ev.author == "story_writer" and ev.content and ev.content.parts
    ]
    state_delta = {
        "current_text": "\n\n".join(passages)[-RECENT_TEXT_CHARS:],
        "instruction": instruction,
    }
    for key in ("premise", "genre", "tone"):
        if getattr(req, key):
            state_delta[key] = getattr(req, key)
    return runner, instruction, state_delta


@router.post("/expand")
async def expand_story_endpoint(req: StoryRequest, request: Request):
    runner, instruction, state_delta = await _story_turn_inputs(req, request)
    expansion = await execute_runner_turn(
        runner,
        user_id="User",
        session_id=req.session_id,
        message=instruction,
        state_delta=state_delta,
        author="story_writer",
    )
    return {"expansion": expansion, "session_id": req.session_id}


@router.post("/expand/stream")
async def expand_story_stream(req: StoryRequest, request: Request):
    runner, instruction, state_delta = await _story_turn_inputs(req, request)
    generator = stream_runner_turn(
        runner,
        user_id="User",
        session_id=req.session_id,
        message=instruction,
        state_delta=state_delta,
        author="story_writer",
    )

    return StreamingResponse(
        format_sse_stream(generator, chunk_size=req.chunk_size),
        media_type="text/event-stream",
        headers={
            "X-Session-ID": req.session_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
