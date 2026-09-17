from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.core.types import StoryRequest
import story_rp_engine.story.workflow as story_workflow

router = APIRouter(prefix="/api/v1/story", tags=["Story Co-Pilot"])


def expand_story(*args, **kwargs):
    return story_workflow.expand_story(*args, **kwargs)


def stream_expand_story(*args, **kwargs):
    return story_workflow.stream_expand_story(*args, **kwargs)


@router.post("/expand")
def expand_story_endpoint(req: StoryRequest, request: Request):
    config = request.app.state.config
    expansion = expand_story(req, config)
    return {"expansion": expansion}


@router.post("/expand/stream")
def expand_story_stream(req: StoryRequest, request: Request):
    config = request.app.state.config
    generator = stream_expand_story(req, config)

    def event_stream():
        for chunk in generator:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
