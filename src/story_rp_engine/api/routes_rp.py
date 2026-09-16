from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from story_rp_engine.core.types import CharacterCardV2, ChatMessage
import story_rp_engine.rp.agent as rp_agent
from story_rp_engine.rp.agent import create_rp_agent, run_rp_turn, stream_rp_turn

_original_run_rp_turn = run_rp_turn
_original_stream_rp_turn = stream_rp_turn
_original_create_rp_agent = create_rp_agent


def _get_run_rp_turn():
    import story_rp_engine.api.routes_rp as self_mod
    if getattr(self_mod, "run_rp_turn", None) is not _original_run_rp_turn:
        return self_mod.run_rp_turn
    if rp_agent.run_rp_turn is not _original_run_rp_turn:
        return rp_agent.run_rp_turn
    return run_rp_turn


def _get_stream_rp_turn():
    import story_rp_engine.api.routes_rp as self_mod
    if getattr(self_mod, "stream_rp_turn", None) is not _original_stream_rp_turn:
        return self_mod.stream_rp_turn
    if rp_agent.stream_rp_turn is not _original_stream_rp_turn:
        return rp_agent.stream_rp_turn
    if rp_agent.run_rp_turn is not _original_run_rp_turn:
        def _adapted_stream(*args, **kwargs):
            val = rp_agent.run_rp_turn(*args, **kwargs)
            words = val.split(" ")
            for i, w in enumerate(words):
                yield w if i == 0 else " " + w
        return _adapted_stream
    return stream_rp_turn


def _get_create_rp_agent():
    import story_rp_engine.api.routes_rp as self_mod
    if getattr(self_mod, "create_rp_agent", None) is not _original_create_rp_agent:
        return self_mod.create_rp_agent
    if rp_agent.create_rp_agent is not _original_create_rp_agent:
        return rp_agent.create_rp_agent
    return create_rp_agent


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

    agent_creator = _get_create_rp_agent()
    agent = agent_creator(card, config, active_lore=[], user_name=req.user_name or "User")
    turn_fn = _get_run_rp_turn()
    reply = turn_fn(agent, history, req.message, authors_note=req.authors_note)

    # Persist updated history
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

    agent_creator = _get_create_rp_agent()
    agent = agent_creator(card, config, active_lore=[], user_name=req.user_name or "User")
    stream_fn = _get_stream_rp_turn()

    def event_stream():
        accumulated_chunks = []
        try:
            for chunk in stream_fn(agent, history, req.message, authors_note=req.authors_note):
                accumulated_chunks.append(chunk)
                yield f"data: {chunk}\n\n"
        finally:
            full_reply = "".join(accumulated_chunks)
            if full_reply:
                updated_history = history + [
                    ChatMessage(role="user", content=req.message),
                    ChatMessage(role="assistant", content=full_reply),
                ]
                try:
                    store.save_history(req.session_id, updated_history)
                except Exception:
                    pass
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
