from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.api.chat_sessions import (
    USER_ID,
    DeleteTurnRequest,
    RenameSessionRequest,
    chat_state_delta,
    checked_session_id,
    delete_chat_turn,
    list_chat_sessions,
    message_event_indexes,
    rename_chat,
)
from story_rp_engine.core.agent_utils import format_sse_stream, stream_group_turn
from story_rp_engine.core.types import GroupCard, GroupChatRequest
from story_rp_engine.group.selector_agent import SPEAKER_SELECTOR
from story_rp_engine.group.workflow import group_speakers
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Group Chat"])

GROUP_APP = "group_app"


@router.post("/groups")
async def save_group(group: GroupCard, request: Request):
    try:
        await request.app.state.store.save_group(group.group_id, group)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "group_id": group.group_id}


@router.get("/groups")
async def list_groups(request: Request):
    return await request.app.state.store.list_groups()


@router.get("/groups/{group_id}")
async def get_group(group_id: str, request: Request):
    try:
        group = await request.app.state.store.get_group(group_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return group


@router.delete("/groups/{group_id}")
async def delete_group(group_id: str, request: Request):
    try:
        clean_id = _sanitize_key(group_id)
        deleted = await request.app.state.store.delete_group(clean_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Group not found")
    request.app.state.agent_registry.forget_group(clean_id)
    return {"status": "deleted", "group_id": clean_id}


@router.post("/group/chat/stream")
async def chat_group_stream(req: GroupChatRequest, request: Request):
    try:
        session_id = _sanitize_key(req.session_id)
        runner = await request.app.state.agent_registry.get_or_create_group_runner(req.group_id)
        state_delta = await chat_state_delta(req, request, group_id=req.group_id)
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e) else 400, detail=str(e))
    group = await request.app.state.store.get_group(req.group_id)

    generator = stream_group_turn(
        runner,
        user_id=USER_ID,
        session_id=session_id,
        message=req.message,
        speakers=group_speakers(group),
        state_delta=state_delta,
    )
    return StreamingResponse(
        format_sse_stream(generator, chunk_size=req.chunk_size),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/group/sessions")
async def list_group_sessions(group_id: str, request: Request):
    """Chats with one group, newest first."""
    return await list_chat_sessions(request.app.state.session_service, GROUP_APP, "group_id", group_id)


@router.get("/group/sessions/{session_id}/turns")
async def get_group_session_turns(session_id: str, request: Request):
    """The chat as the UI shows it, without the speaker selector's events. Each turn names its speaker's char_id
    (None for the user; the raw agent name for a member who has since left the group)."""
    session = await request.app.state.session_service.get_session(
        app_name=GROUP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    if not session:
        return {"turns": []}
    group = await request.app.state.store.get_group(session.state["group_id"])
    speakers = group_speakers(group) if group else {}
    turns = []
    for pos, idx in enumerate(message_event_indexes(session.events, skip_authors={SPEAKER_SELECTOR})):
        ev = session.events[idx]
        turns.append({
            "index": pos,
            "role": ev.content.role,
            "speaker": None if ev.author == "user" else speakers.get(ev.author, ev.author),
            "text": "".join(p.text for p in ev.content.parts if p.text),
        })
    return {"turns": turns}


@router.patch("/group/sessions/{session_id}")
async def rename_group_session(session_id: str, req: RenameSessionRequest, request: Request):
    await rename_chat(request.app.state.session_service, GROUP_APP, session_id, req.title)
    return {"status": "renamed", "session_id": session_id}


@router.delete("/group/sessions/{session_id}")
async def delete_group_session(session_id: str, request: Request):
    await request.app.state.session_service.delete_session(
        app_name=GROUP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    return {"status": "deleted", "session_id": session_id}


@router.post("/group/sessions/{session_id}/turns/delete")
async def delete_group_session_turn(session_id: str, req: DeleteTurnRequest, request: Request):
    remaining = await delete_chat_turn(
        request.app.state.session_service,
        GROUP_APP,
        checked_session_id(session_id),
        req.turn_index,
        req.truncate_subsequent,
        skip_authors={SPEAKER_SELECTOR},
    )
    return {"status": "ok", "remaining_turns": remaining}
