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
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    format_sse_stream,
    stream_runner_turn,
)
from story_rp_engine.core.types import CharacterCard, RPChatRequest
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Roleplay"])

RP_APP = "rp_app"


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
        state_delta = await chat_state_delta(req, request, char_id=req.char_id, greeting=req.greeting)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    reply = await execute_runner_turn(
        runner,
        user_id=USER_ID,
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
        state_delta = await chat_state_delta(req, request, char_id=req.char_id, greeting=req.greeting)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    generator = stream_runner_turn(
        runner,
        user_id=USER_ID,
        session_id=req.session_id,
        message=req.message,
        state_delta=state_delta,
    )

    return StreamingResponse(
        format_sse_stream(generator, chunk_size=req.chunk_size, config=request.app.state.config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/rp/sessions")
async def list_rp_sessions(char_id: str, request: Request):
    """Chats with one character, newest first."""
    return await list_chat_sessions(request.app.state.session_service, RP_APP, "char_id", char_id)


@router.get("/rp/sessions/{session_id}/turns")
async def get_session_turns(session_id: str, request: Request):
    session = await request.app.state.session_service.get_session(
        app_name=RP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    if not session:
        return {"turns": []}
    turns = []
    for pos, idx in enumerate(message_event_indexes(session.events)):
        content = session.events[idx].content
        turns.append({"index": pos, "role": content.role, "text": "".join(p.text for p in content.parts if p.text)})
    return {"turns": turns}


@router.patch("/rp/sessions/{session_id}")
async def rename_rp_session(session_id: str, req: RenameSessionRequest, request: Request):
    await rename_chat(request.app.state.session_service, RP_APP, session_id, req.title)
    return {"status": "renamed", "session_id": session_id}


@router.delete("/rp/sessions/{session_id}")
async def clear_session(session_id: str, request: Request):
    await request.app.state.session_service.delete_session(
        app_name=RP_APP, user_id=USER_ID, session_id=checked_session_id(session_id)
    )
    return {"status": "deleted", "session_id": session_id}


@router.post("/rp/sessions/{session_id}/turns/delete")
async def delete_session_turn(session_id: str, req: DeleteTurnRequest, request: Request):
    remaining = await delete_chat_turn(
        request.app.state.session_service,
        RP_APP,
        checked_session_id(session_id),
        req.turn_index,
        req.truncate_subsequent,
    )
    return {"status": "ok", "remaining_turns": remaining}

