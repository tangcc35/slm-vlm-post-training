"""Chat session helpers shared by the roleplay and group chat routes; each router passes its own ADK app name."""

from typing import Collection, List
from fastapi import HTTPException, Request
from google.adk.events import Event, EventActions
from pydantic import BaseModel
from story_rp_engine.storage.store import _sanitize_key

# Every chat session uses this user ID; the display name lives in session state.
USER_ID = "User"


class RenameSessionRequest(BaseModel):
    title: str


class DeleteTurnRequest(BaseModel):
    turn_index: int
    truncate_subsequent: bool = False


def checked_session_id(session_id: str) -> str:
    try:
        return _sanitize_key(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def chat_state_delta(req, request: Request, **fields) -> dict:
    """Session state changes for a chat turn; a lorebook is copied in only when lorebook_id is sent.

    `fields` adds the chat's own keys (char_id and greeting, or group_id). The owner ID and last_message are
    read by the history list, which gets session state but no events.
    """
    state_delta = {
        "authors_note": req.authors_note,
        "user_name": req.user_name or "User",
        **fields,
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


def message_event_indexes(events, skip_authors: Collection[str] = ()) -> List[int]:
    """Indexes of the events the chat shows: user and model text, without ADK compaction summaries or the events
    of skip_authors.

    Compaction only adds a summary event and keeps the raw events, so the full history is still there.
    """
    return [
        i
        for i, ev in enumerate(events)
        if not (ev.actions and ev.actions.compaction)
        and ev.author not in skip_authors
        and ev.content
        and any(p.text for p in ev.content.parts or [])
    ]


async def list_chat_sessions(session_service, app_name: str, owner_key: str, owner_id: str) -> list:
    """Chats whose state[owner_key] is owner_id (e.g. a character's chats), newest first."""
    response = await session_service.list_sessions(app_name=app_name, user_id=USER_ID)
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
        if s.state.get(owner_key) == owner_id
    ]


async def rename_chat(session_service, app_name: str, session_id: str, title: str) -> None:
    session = await session_service.get_session(app_name=app_name, user_id=USER_ID, session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # A state-only event: ADK has no other way to update session state, and having no content
    # keeps it out of the chat and the prompt.
    await session_service.append_event(
        session, Event(author="user", actions=EventActions(state_delta={"title": title.strip()}))
    )


async def delete_chat_turn(
    session_service,
    app_name: str,
    session_id: str,
    turn_index: int,
    truncate_subsequent: bool,
    skip_authors: Collection[str] = (),
) -> int:
    """Deletes one shown message, or it and everything after it, by rebuilding the session; returns the number
    of events left.

    turn_index counts the messages the chat shows (see message_event_indexes), not raw events.
    """
    session = await session_service.get_session(app_name=app_name, user_id=USER_ID, session_id=session_id)
    if not session:
        return 0

    events = list(session.events)
    message_indexes = message_event_indexes(events, skip_authors)
    if truncate_subsequent:
        if 0 <= turn_index < len(message_indexes):
            events = events[: message_indexes[turn_index]]
        elif turn_index < 0:
            events = []
    elif 0 <= turn_index < len(message_indexes):
        events.pop(message_indexes[turn_index])

    await session_service.delete_session(app_name=app_name, user_id=USER_ID, session_id=session_id)
    new_session = await session_service.create_session(
        app_name=app_name, user_id=USER_ID, session_id=session_id, state=session.state
    )
    for ev in events:
        await session_service.append_event(new_session, ev)
    return len(events)
