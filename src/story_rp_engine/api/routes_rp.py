import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    format_sse_stream,
    stream_runner_turn,
)
from story_rp_engine.core.types import CharacterCard, RPChatRequest
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Roleplay"])


@router.post("/characters")
def save_character(card: CharacterCard, request: Request):
    if not card.char_id:
        raise HTTPException(status_code=400, detail="char_id is required in request body")
    store = request.app.state.store
    try:
        store.save_character(card.char_id, card)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "char_id": card.char_id}



@router.get("/characters")
def list_characters(request: Request):
    store = request.app.state.store
    return store.list_characters()


@router.get("/characters/{char_id}")
def get_character(char_id: str, request: Request):
    store = request.app.state.store
    try:
        card = store.get_character(char_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not card:
        raise HTTPException(status_code=404, detail="Character not found")
    return card


@router.delete("/characters/{char_id}")
def delete_character(char_id: str, request: Request):
    store = request.app.state.store
    registry = getattr(request.app.state, "agent_registry", None)
    try:
        clean_id = _sanitize_key(char_id)
        path = os.path.join(store.char_dir, f"{clean_id}.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Character not found")
        os.remove(path)
        if registry:
            registry._rp_agents.pop(clean_id, None)
            registry._rp_runners.pop(clean_id, None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted", "char_id": clean_id}


@router.post("/rp/chat")
async def chat_rp(req: RPChatRequest, request: Request):
    registry = request.app.state.agent_registry

    try:
        _sanitize_key(req.session_id)
        runner = registry.get_or_create_rp_runner(req.char_id)
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
        runner = registry.get_or_create_rp_runner(req.char_id)
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
    )
