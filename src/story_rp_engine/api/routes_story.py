from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from pydantic import BaseModel
from story_rp_engine.api.chat_sessions import load_lorebooks
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
    if req.lorebook_ids:
        state_delta["lorebook"] = await load_lorebooks(request, req.lorebook_ids)
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


@router.get("/sessions")
async def list_story_sessions(request: Request):
    """Stories, newest first."""
    runner = _get_story_runner(request)
    response = await runner.session_service.list_sessions(app_name=runner.app_name, user_id="User")
    return [
        {
            "session_id": s.id,
            "updated_at": s.last_update_time,
            "title": s.state.get("title"),
            "premise": s.state.get("premise", ""),
            "genre": s.state.get("genre"),
            "tone": s.state.get("tone"),
        }
        for s in sorted(response.sessions, key=lambda s: s.last_update_time, reverse=True)
    ]


@router.get("/sessions/{session_id}/messages")
async def get_story_messages(session_id: str, request: Request):
    """The chat as the UI shows it: the user's messages and the writer's replies, without the director's notes
    or compaction summaries (compaction keeps the raw events)."""
    runner = _get_story_runner(request)
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id="User", session_id=session_id
    )
    if not session:
        raise HTTPException(status_code=404, detail="Story not found")
    messages = []
    for ev in session.events:
        if ev.author not in ("user", "story_writer") or not ev.content or not ev.content.parts:
            continue
        text = "".join(p.text for p in ev.content.parts if p.text)
        if text:
            messages.append({"role": "user" if ev.author == "user" else "assistant", "content": text})
    setup = {key: session.state.get(key) for key in ("premise", "genre", "tone")}
    return {"messages": messages, "setup": setup}


class RenameSessionRequest(BaseModel):
    title: str


@router.patch("/sessions/{session_id}")
async def rename_story_session(session_id: str, req: RenameSessionRequest, request: Request):
    runner = _get_story_runner(request)
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id="User", session_id=session_id
    )
    if not session:
        raise HTTPException(status_code=404, detail="Story not found")
    # A state-only event (see routes_rp.rename_rp_session).
    await runner.session_service.append_event(
        session, Event(author="user", actions=EventActions(state_delta={"title": req.title.strip()}))
    )
    return {"status": "renamed", "session_id": session_id}


@router.delete("/sessions/{session_id}")
async def delete_story_session(session_id: str, request: Request):
    runner = _get_story_runner(request)
    await runner.session_service.delete_session(app_name=runner.app_name, user_id="User", session_id=session_id)
    return {"status": "deleted", "session_id": session_id}
