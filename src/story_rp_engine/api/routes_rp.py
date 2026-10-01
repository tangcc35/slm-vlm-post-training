import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    format_sse_stream,
    stream_runner_turn,
)
from story_rp_engine.core.types import CharacterCard, RPChatRequest
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Roleplay"])


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
        await store.delete_character(clean_id)
        if registry:
            registry._rp_agents.pop(clean_id, None)
            registry._rp_runners.pop(clean_id, None)
    except KeyError:
        raise HTTPException(status_code=404, detail="Character not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted", "char_id": clean_id}


@router.post("/rp/chat")
async def chat_rp(req: RPChatRequest, request: Request):
    registry = request.app.state.agent_registry

    try:
        _sanitize_key(req.session_id)
        runner = await registry.get_or_create_rp_runner_async(req.char_id)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    reply = await execute_runner_turn(
        runner,
        user_id=req.user_name or "User",
        session_id=req.session_id,
        message=req.message,
        state_delta={"authors_note": req.authors_note},
    )

    return {"reply": reply, "session_id": req.session_id}


@router.post("/rp/chat/stream")
async def chat_rp_stream(req: RPChatRequest, request: Request):
    registry = request.app.state.agent_registry

    try:
        _sanitize_key(req.session_id)
        runner = await registry.get_or_create_rp_runner_async(req.char_id)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    generator = stream_runner_turn(
        runner,
        user_id=req.user_name or "User",
        session_id=req.session_id,
        message=req.message,
        state_delta={"authors_note": req.authors_note},
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


class DeleteTurnRequest(BaseModel):
    turn_index: int
    truncate_subsequent: bool = False


@router.get("/rp/sessions/{session_id}/turns")
async def get_session_turns(session_id: str, request: Request):
    try:
        _sanitize_key(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session_service = request.app.state.session_service
    session = await session_service.get_session(app_name="rp_app", user_id="User", session_id=session_id)
    if not session or not session.events:
        return {"turns": []}

    turns = []
    for idx, ev in enumerate(session.events):
        text = ""
        is_compaction = bool(ev.actions and ev.actions.compaction)
        if is_compaction and ev.actions.compaction.compacted_content:
            parts = getattr(ev.actions.compaction.compacted_content, "parts", None) or []
            text = "".join(p.text for p in parts if getattr(p, "text", None))
            role = "compaction"
        elif ev.content and ev.content.parts:
            text = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
            role = getattr(ev.content, "role", "unknown") if ev.content else "system"
        else:
            role = "system"

        if text:
            turn_data = {"index": idx, "role": role, "text": text}
            if is_compaction:
                turn_data["is_compaction"] = True
            turns.append(turn_data)
    return {"turns": turns}


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

    # Recreate session with pruned events
    events = list(session.events)
    if req.truncate_subsequent:
        if req.turn_index >= 0:
            events = events[:req.turn_index]
        else:
            events = []
    else:
        if 0 <= req.turn_index < len(events):
            events.pop(req.turn_index)

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

