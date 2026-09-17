from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.core.agent_utils import (
    extract_agent_response_text,
    stream_agent_response,
)
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.workflow import prepare_story_expansion

router = APIRouter(prefix="/api/v1/story", tags=["Story Co-Pilot"])


@router.post("/expand")
def expand_story_endpoint(req: StoryRequest, request: Request):
    config = request.app.state.config
    writer, prompt = prepare_story_expansion(req, config)
    expansion = extract_agent_response_text(writer.invoke(prompt))
    return {"expansion": expansion}


@router.post("/expand/stream")
def expand_story_stream(req: StoryRequest, request: Request):
    config = request.app.state.config
    writer, prompt = prepare_story_expansion(req, config)
    generator = stream_agent_response(writer, prompt)

    def event_stream():
        for chunk in generator:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
