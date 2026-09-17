import uuid
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from google.adk import Runner
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    stream_runner_turn,
)
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.workflow import format_story_input

router = APIRouter(prefix="/api/v1/story", tags=["Story Co-Pilot"])


@router.post("/expand")
async def expand_story_endpoint(req: StoryRequest, request: Request):
    session_service = request.app.state.session_service
    workflow = request.app.state.agent_registry.get_story_workflow()
    runner = Runner(
        agent=workflow,
        session_service=session_service,
        app_name="story_app",
        auto_create_session=True,
    )
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
    session_service = request.app.state.session_service
    workflow = request.app.state.agent_registry.get_story_workflow()
    runner = Runner(
        agent=workflow,
        session_service=session_service,
        app_name="story_app",
        auto_create_session=True,
    )
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
