from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.core.types import StoryRequest
import story_rp_engine.story.workflow as story_workflow
from story_rp_engine.story.workflow import expand_story, stream_expand_story

_original_expand_story = expand_story
_original_stream_expand_story = stream_expand_story


def _get_expand_story():
    import story_rp_engine.api.routes_story as self_mod
    if getattr(self_mod, "expand_story", None) is not _original_expand_story:
        return self_mod.expand_story
    if story_workflow.expand_story is not _original_expand_story:
        return story_workflow.expand_story
    return expand_story


def _get_stream_expand_story():
    import story_rp_engine.api.routes_story as self_mod
    if getattr(self_mod, "stream_expand_story", None) is not _original_stream_expand_story:
        return self_mod.stream_expand_story
    if story_workflow.stream_expand_story is not _original_stream_expand_story:
        return story_workflow.stream_expand_story
    if story_workflow.expand_story is not _original_expand_story:
        def _adapted_stream(*args, **kwargs):
            val = story_workflow.expand_story(*args, **kwargs)
            words = val.split(" ")
            for i, w in enumerate(words):
                yield w if i == 0 else " " + w
        return _adapted_stream
    return stream_expand_story


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
    stream_fn = _get_stream_expand_story()

    def event_stream():
        for chunk in stream_fn(req, config):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


