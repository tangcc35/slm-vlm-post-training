from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from story_rp_engine.core.agent_utils import (
    extract_agent_response_text,
    stream_agent_response,
)
from story_rp_engine.core.types import CharacterCardV2, ChatMessage
from story_rp_engine.rp.agent import build_rp_turn_prompt, create_rp_agent

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
def chat_rp(req: RPChatRequest, request: Request):
    store = request.app.state.store
    config = request.app.state.config

    try:
        card = store.get_character(req.char_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not card:
        raise HTTPException(status_code=404, detail="Character not found")

    try:
        history = store.get_history(req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    agent = create_rp_agent(card, config, active_lore=[], user_name=req.user_name or "User")
    prompt = build_rp_turn_prompt(history, req.message, authors_note=req.authors_note)
    reply = extract_agent_response_text(agent.invoke(prompt))

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
def chat_rp_stream(req: RPChatRequest, request: Request):
    store = request.app.state.store
    config = request.app.state.config

    try:
        card = store.get_character(req.char_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not card:
        raise HTTPException(status_code=404, detail="Character not found")

    try:
        history = store.get_history(req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    agent = create_rp_agent(card, config, active_lore=[], user_name=req.user_name or "User")
    prompt = build_rp_turn_prompt(history, req.message, authors_note=req.authors_note)
    generator = stream_agent_response(agent, prompt)


    accumulated_chunks = []

    def event_stream():
        try:
            for chunk in generator:
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
