from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.adk.events import Event, EventActions
from pydantic import BaseModel
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    format_sse_stream,
    stream_runner_turn,
)
from story_rp_engine.core.types import CharacterCard, RPChatRequest
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Roleplay"])


async def _rp_state_delta(req: RPChatRequest, request: Request) -> dict:
    """Session state changes for a chat turn; a lorebook is copied in only when lorebook_id is sent."""
    state_delta = {
        "authors_note": req.authors_note,
        "user_name": req.user_name or "User",
        "greeting": req.greeting,
        # Read by the history list, which gets session state but no events.
        "char_id": req.char_id,
        "last_message": req.message[:80],
    }
    if req.lorebook_id is not None:
        lorebook = None
        if req.lorebook_id:
            lorebook = await request.app.state.store.get_lorebook(req.lorebook_id)
            if lorebook is None:
                raise HTTPException(status_code=404, detail="Lorebook not found")
        state_delta["lorebook"] = lorebook.model_dump() if lorebook else None
        # Read by the history list, so reopening a chat reselects its lorebook.
        state_delta["lorebook_id"] = req.lorebook_id or None
    return state_delta


@router.post("/characters")
async def save_character(card: CharacterCard, request: Request):
    if not card.char_id:
        raise HTTPException(status_code=400, detail="char_id is required in request body")
    store = request.app.state.store
    try:
        await store.save_character(card.char_id, card)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "char_id": card.char_id}



@router.get("/characters")
async def list_characters(request: Request):
    store = request.app.state.store
    return await store.list_characters()


@router.get("/characters/{char_id}")
async def get_character(char_id: str, request: Request):
    store = request.app.state.store
    try:
        card = await store.get_character(char_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not card:
        raise HTTPException(status_code=404, detail="Character not found")
    return card


@router.delete("/characters/{char_id}")
async def delete_character(char_id: str, request: Request):
    store = request.app.state.store
    registry = getattr(request.app.state, "agent_registry", None)
    try:
        clean_id = _sanitize_key(char_id)
        deleted = await store.delete_character(clean_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Character not found")
    if registry:
        registry.forget_rp_agent(clean_id)
    return {"status": "deleted", "char_id": clean_id}


@router.post("/rp/chat")
async def chat_rp(req: RPChatRequest, request: Request):
    registry = request.app.state.agent_registry

    try:
        _sanitize_key(req.session_id)
        runner = await registry.get_or_create_rp_runner(req.char_id)
        state_delta = await _rp_state_delta(req, request)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    reply = await execute_runner_turn(
        runner,
        user_id="User",  # matches the session endpoints; the display name lives in state
        session_id=req.session_id,
        message=req.message,
        state_delta=state_delta,
    )

    return {"reply": reply, "session_id": req.session_id}


@router.post("/rp/chat/stream")
async def chat_rp_stream(req: RPChatRequest, request: Request):
    registry = request.app.state.agent_registry

    try:
        _sanitize_key(req.session_id)
        runner = await registry.get_or_create_rp_runner(req.char_id)
        state_delta = await _rp_state_delta(req, request)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    generator = stream_runner_turn(
        runner,
        user_id="User",  # matches the session endpoints; the display name lives in state
        session_id=req.session_id,
        message=req.message,
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


class RenameSessionRequest(BaseModel):
    title: str


class DeleteTurnRequest(BaseModel):
    turn_index: int
    truncate_subsequent: bool = False


def _message_event_indexes(events) -> list[int]:
    """Indexes of the events the chat shows: user and model text, without ADK compaction summaries.

    Compaction only adds a summary event and keeps the raw events, so the full history is still there.
    """
    return [
        i
        for i, ev in enumerate(events)
        if not (ev.actions and ev.actions.compaction)
        and ev.content
        and any(p.text for p in ev.content.parts or [])
    ]


@router.get("/rp/sessions")
async def list_rp_sessions(char_id: str, request: Request):
    """Chats with one character, newest first."""
    response = await request.app.state.session_service.list_sessions(app_name="rp_app", user_id="User")
    return [
        {
            "session_id": s.id,
            "updated_at": s.last_update_time,
            "title": s.state.get("title"),
            "last_message": s.state.get("last_message", ""),
            "greeting": s.state.get("greeting"),
            "user_name": s.state.get("user_name"),
            "authors_note": s.state.get("authors_note"),
            "lorebook_id": s.state.get("lorebook_id"),
        }
        for s in sorted(response.sessions, key=lambda s: s.last_update_time, reverse=True)
        if s.state.get("char_id") == char_id
    ]


@router.get("/rp/sessions/{session_id}/turns")
async def get_session_turns(session_id: str, request: Request):
    try:
        _sanitize_key(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session_service = request.app.state.session_service
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id=session_id)
    if not session:
        return {"turns": []}
    turns = []
    for pos, idx in enumerate(_message_event_indexes(session.events)):
        content = session.events[idx].content
        turns.append({"index": pos, "role": content.role, "text": "".join(p.text for p in content.parts if p.text)})
    return {"turns": turns}


@router.patch("/rp/sessions/{session_id}")
async def rename_rp_session(session_id: str, req: RenameSessionRequest, request: Request):
    session_service = request.app.state.session_service
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # A state-only event: ADK has no other way to update session state, and having no content
    # keeps it out of the chat and the prompt.
    await session_service.append_event(
        session, Event(author="user", actions=EventActions(state_delta={"title": req.title.strip()}))
    )
    return {"status": "renamed", "session_id": session_id}


@router.delete("/rp/sessions/{session_id}")
async def clear_session(session_id: str, request: Request):
    try:
        _sanitize_key(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session_service = request.app.state.session_service
    await session_service.delete_session(app_name="rp_app", user_id="User", session_id=session_id)
    return {"status": "deleted", "session_id": session_id}


@router.post("/rp/sessions/{session_id}/turns/delete")
async def delete_session_turn(session_id: str, req: DeleteTurnRequest, request: Request):
    try:
        _sanitize_key(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session_service = request.app.state.session_service
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id=session_id)
    if not session:
        return {"status": "ok", "remaining_turns": 0}

    # turn_index counts the messages the chat shows (see get_session_turns), not raw events.
    events = list(session.events)
    message_indexes = _message_event_indexes(events)
    if req.truncate_subsequent:
        if 0 <= req.turn_index < len(message_indexes):
            events = events[: message_indexes[req.turn_index]]
        elif req.turn_index < 0:
            events = []
    elif 0 <= req.turn_index < len(message_indexes):
        events.pop(message_indexes[req.turn_index])

    await session_service.delete_session(app_name="rp_app", user_id="User", session_id=session_id)
    new_session = await session_service.create_session(
        app_name="rp_app",
        user_id="User",
        session_id=session_id,
        state=session.state,
    )
    for ev in events:
        await session_service.append_event(new_session, ev)

    return {"status": "ok", "remaining_turns": len(events)}

