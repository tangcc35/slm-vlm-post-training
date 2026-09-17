from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.adk import Runner
from pydantic import BaseModel
from story_rp_engine.core.agent_utils import (
    execute_runner_turn,
    stream_runner_turn,
)
from story_rp_engine.core.types import CharacterCardV2, ChatMessage

router = APIRouter(prefix="/api/v1", tags=["Roleplay"])


class RPChatRequest(BaseModel):
    char_id: str
    session_id: str
    message: str
    authors_note: Optional[str] = None
    user_name: Optional[str] = "User"


@router.post("/characters")
def save_character(char_id: str, card: CharacterCardV2, request: Request):
    store = request.app.state.store
    try:
        store.save_character(char_id, card)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "char_id": char_id}


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


@router.post("/rp/chat")
async def chat_rp(req: RPChatRequest, request: Request):
    store = request.app.state.store
    session_service = request.app.state.session_service
    registry = request.app.state.agent_registry

    try:
        history = store.get_history(req.session_id)
        agent = registry.get_or_create_rp_agent(req.char_id)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name="rp_app",
        auto_create_session=True,
    )

    reply = await execute_runner_turn(
        runner,
        user_id=req.user_name or "User",
        session_id=req.session_id,
        message=req.message,
        state_delta={"authors_note": req.authors_note},
    )

    updated_history = history + [
        ChatMessage(role="user", content=req.message),
        ChatMessage(role="assistant", content=reply),
    ]
    try:
        store.save_history(req.session_id, updated_history)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"reply": reply, "session_id": req.session_id}


@router.post("/rp/chat/stream")
async def chat_rp_stream(req: RPChatRequest, request: Request):
    store = request.app.state.store
    session_service = request.app.state.session_service
    registry = request.app.state.agent_registry

    try:
        history = store.get_history(req.session_id)
        agent = registry.get_or_create_rp_agent(req.char_id)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Character not found")
        raise HTTPException(status_code=400, detail=str(e))

    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name="rp_app",
        auto_create_session=True,
    )

    generator = stream_runner_turn(
        runner,
        user_id=req.user_name or "User",
        session_id=req.session_id,
        message=req.message,
        state_delta={"authors_note": req.authors_note},
    )

    accumulated_chunks = []

    async def event_stream():
        try:
            async for chunk in generator:
                accumulated_chunks.append(chunk)
                yield f"data: {chunk}\n\n"
        finally:
            complete_reply = "".join(accumulated_chunks).strip()
            updated_history = history + [
                ChatMessage(role="user", content=req.message),
                ChatMessage(role="assistant", content=complete_reply),
            ]
            try:
                store.save_history(req.session_id, updated_history)
            except ValueError:
                pass
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
