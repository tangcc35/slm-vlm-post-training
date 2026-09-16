from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from story_rp_engine.core.types import CharacterCardV2, ChatMessage
from story_rp_engine.rp.agent import create_rp_agent, run_rp_turn

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
    store.save_character(char_id, card)
    return {"status": "saved", "char_id": char_id}


@router.get("/characters")
def list_characters(request: Request):
    store = request.app.state.store
    return store.list_characters()


@router.get("/characters/{char_id}")
def get_character(char_id: str, request: Request):
    store = request.app.state.store
    card = store.get_character(char_id)
    if not card:
        raise HTTPException(status_code=404, detail="Character not found")
    return card


@router.post("/rp/chat")
def chat_rp(req: RPChatRequest, request: Request):
    store = request.app.state.store
    config = request.app.state.config

    card = store.get_character(req.char_id)
    if not card:
        raise HTTPException(status_code=404, detail="Character not found")

    history = store.get_history(req.session_id)
    agent = create_rp_agent(card, config, active_lore=[], user_name=req.user_name or "User")
    reply = run_rp_turn(agent, history, req.message, authors_note=req.authors_note)

    # Persist updated history
    updated_history = history + [
        ChatMessage(role="user", content=req.message),
        ChatMessage(role="assistant", content=reply),
    ]
    store.save_history(req.session_id, updated_history)

    return {"reply": reply, "session_id": req.session_id}


@router.post("/rp/chat/stream")
def chat_rp_stream(req: RPChatRequest, request: Request):
    # Streaming SSE generator wrapper
    res = chat_rp(req, request)

    def event_stream():
        yield f"data: {res['reply']}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
