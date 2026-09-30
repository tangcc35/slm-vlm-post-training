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


@router.post("/expand")
async def expand_story_endpoint(req: StoryRequest, request: Request):
    if not req.session_id or not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    runner = _get_story_runner(request)
    state_delta = {
        "premise": req.premise or "Not specified",
        "genre": req.genre or "Fiction",
        "tone": req.tone or "Balanced",
        "current_text": req.current_text or "",
        "instruction": req.instruction or "Expand the story based on the context.",
    }
    user_instruction = req.instruction or "Expand the story based on the context."
    expansion = await execute_runner_turn(
        runner,
        user_id="User",
        session_id=req.session_id,
        message=user_instruction,
        state_delta=state_delta,
    )
    return {"expansion": expansion, "session_id": req.session_id}


@router.post("/expand/stream")
async def expand_story_stream(req: StoryRequest, request: Request):
    if not req.session_id or not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    runner = _get_story_runner(request)
    state_delta = {
        "premise": req.premise or "Not specified",
        "genre": req.genre or "Fiction",
        "tone": req.tone or "Balanced",
        "current_text": req.current_text or "",
        "instruction": req.instruction or "Expand the story based on the context.",
    }
    user_instruction = req.instruction or "Expand the story based on the context."
    generator = stream_runner_turn(
        runner,
        user_id="User",
        session_id=req.session_id,
        message=user_instruction,
        state_delta=state_delta,
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
