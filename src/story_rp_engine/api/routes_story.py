from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.core.types import StoryRequest
import story_rp_engine.story.workflow as story_workflow
from story_rp_engine.story.workflow import expand_story

_original_expand_story = expand_story


def _get_expand_story():
    if story_workflow.expand_story is not _original_expand_story:
        return story_workflow.expand_story
    return expand_story


router = APIRouter(prefix="/api/v1/story", tags=["Story Co-Pilot"])


@router.post("/expand")
def expand_story_endpoint(req: StoryRequest, request: Request):
    config = request.app.state.config
    fn = _get_expand_story()
    expansion = fn(req, config)
    return {"expansion": expansion}


@router.post("/expand/stream")
def expand_story_stream(req: StoryRequest, request: Request):
    config = request.app.state.config
    fn = _get_expand_story()
    expansion = fn(req, config)

    def event_stream():
        yield f"data: {expansion}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

