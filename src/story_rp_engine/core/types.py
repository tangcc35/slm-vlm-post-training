from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class CharacterCardV2Data(BaseModel):
    name: str
    description: str
    personality: str
    scenario: str
    first_mes: str
    mes_example: str
    system_prompt: Optional[str] = None
    post_history_instructions: Optional[str] = None
    alternate_greetings: List[str] = Field(default_factory=list)
    creator_notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class CharacterCardV2(BaseModel):
    spec: Literal["chara_card_v2"] = "chara_card_v2"
    spec_version: Literal["2.0"] = "2.0"
    data: CharacterCardV2Data

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
    premise: Optional[str] = None
    current_text: str = ""
    instruction: Optional[str] = "Continue the story naturally from the current point."
    genre: Optional[str] = "Fiction"
    tone: Optional[str] = "Balanced"
    max_tokens: int = Field(default=512, ge=1, le=4096)
