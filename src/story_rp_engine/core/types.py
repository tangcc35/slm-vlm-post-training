from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class CharacterCard(BaseModel):
    char_id: str
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    system_prompt: Optional[str] = None
    post_history_instructions: Optional[str] = None
    alternate_greetings: List[str] = Field(default_factory=list)
    creator_notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    lorebook_ids: List[str] = Field(default_factory=list, description="Lorebooks selected by default when a new chat with this character starts.")


class LorebookEntry(BaseModel):
    keys: List[str]
    content: str
    insertion_order: int = 100
    enabled: bool = True


class Lorebook(BaseModel):
    name: str
    description: Optional[str] = None
    entries: List[LorebookEntry] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StoryRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session ID for tracking multi-turn story state.")
    # Story setup: sent with the first turn and kept in session state; later turns may leave these out.
    premise: Optional[str] = None
    genre: Optional[str] = None
    tone: Optional[str] = None
    lorebook_ids: List[str] = Field(default_factory=list, description="Copies these lorebooks into the session; like the other setup fields, sent with the first turn.")
    instruction: Optional[str] = "Continue the story naturally from the current point."
    max_tokens: int = Field(default=512, ge=1, le=131072)
    chunk_size: Optional[int] = Field(default=16, ge=1, le=100, description="Number of tokens to buffer before yielding in streaming mode.")


class RPChatRequest(BaseModel):
    char_id: str
    session_id: str
    message: str
    authors_note: Optional[str] = None
    lorebook_ids: Optional[List[str]] = Field(default=None, description="Loads these lorebooks into the session; omit to keep the current ones, empty list to clear them.")
    user_name: Optional[str] = "User"
    greeting: Optional[str] = Field(default=None, description="Opening message shown to the user before their first message.")
    chunk_size: Optional[int] = Field(default=16, ge=1, le=100, description="Number of tokens to buffer before yielding in streaming mode.")


class GroupCard(BaseModel):
    group_id: str
    name: str
    char_ids: List[str] = Field(default_factory=list, description="Members, in the order they speak when the speaker selector picks no one.")
    scenario: str = Field(default="", description="The shared scene; replaces each member card's own scenario.")
    lorebook_ids: List[str] = Field(default_factory=list, description="Lorebooks selected by default when a new chat with this group starts.")


class GroupChatRequest(BaseModel):
    group_id: str
    session_id: str
    message: str
    authors_note: Optional[str] = None
    lorebook_ids: Optional[List[str]] = Field(default=None, description="Loads these lorebooks into the session; omit to keep the current ones, empty list to clear them.")
    user_name: Optional[str] = "User"
    chunk_size: Optional[int] = Field(default=16, ge=1, le=100, description="Number of tokens to buffer before yielding in streaming mode.")

