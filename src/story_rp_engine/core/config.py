import os
from typing import Optional
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    model_name: str = Field(
        default_factory=lambda: os.getenv("STORY_RP_MODEL", os.getenv("MODEL_NAME", "ollama/llama3.1:8b")),
        description="Model identifier for LiteLLM",
    )
    api_base: Optional[str] = Field(
        default_factory=lambda: os.getenv("STORY_RP_API_BASE", os.getenv("API_BASE")),
        description="Custom endpoint URL e.g. http://localhost:11434",
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("STORY_RP_API_KEY", os.getenv("OPENAI_API_KEY")),
        description="API key if required (e.g. for Gemini/OpenAI)",
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("STORY_RP_TEMPERATURE", "0.8")),
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default_factory=lambda: float(os.getenv("STORY_RP_TOP_P", "0.9")),
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("STORY_RP_MAX_TOKENS", "512")),
        ge=1,
        le=8192,
    )
