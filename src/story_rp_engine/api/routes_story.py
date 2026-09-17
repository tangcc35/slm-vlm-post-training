from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.workflow import expand_story, stream_expand_story

router = APIRouter(prefix="/api/v1/story", tags=["Story Co-Pilot"])



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
