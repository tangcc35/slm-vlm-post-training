import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.adk.runners import Runner
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    stream_runner_turn,
)
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.workflow import format_story_input

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
    runner = _get_story_runner(request)
    prompt = format_story_input(req)
    expansion = await execute_runner_turn(
        runner,
        user_id="User",
        session_id=f"story_{uuid.uuid4().hex}",
        message=prompt,
    )
    return {"expansion": expansion}


@router.post("/expand/stream")
async def expand_story_stream(req: StoryRequest, request: Request):
    runner = _get_story_runner(request)
    prompt = format_story_input(req)
    generator = stream_runner_turn(
        runner,
        user_id="User",
        session_id=f"story_{uuid.uuid4().hex}",
        message=prompt,
    )

    async def event_stream():
        async for chunk in generator:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
