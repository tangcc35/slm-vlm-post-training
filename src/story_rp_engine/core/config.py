import os
from typing import Optional
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    model_name: str = Field(
        default_factory=lambda: os.getenv("STORY_RP_MODEL", "ollama/llama3.1:8b"),
        description="Model identifier for LiteLLM",
    )
    api_base: Optional[str] = Field(
        default_factory=lambda: os.getenv("STORY_RP_API_BASE"),
        description="Custom endpoint URL e.g. http://localhost:11434",
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("STORY_RP_API_KEY"),
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
        default_factory=lambda: int(os.getenv("STORY_RP_MAX_TOKENS", "131072")),
        ge=1,
        le=131072,
    )
    storage_dir: str = Field(
        default_factory=lambda: os.getenv("STORY_RP_STORAGE_DIR", ".engine_data"),
        description="Directory for local storage (characters, lorebooks, sessions)",
    )
    db_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("STORY_RP_DB_URL"),
        description="Database connection URL for ADK DatabaseSessionService",
    )
    phoenix_enabled: bool = Field(
        default_factory=lambda: os.getenv("PHOENIX_ENABLED", "0").lower() in ("1", "true", "yes"),
        description="Whether to enable Arize Phoenix OpenTelemetry tracing",
    )
    phoenix_endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            f"http://{'127.0.0.1' if os.getenv('PHOENIX_HOST') in ('0.0.0.0', None, '') else os.getenv('PHOENIX_HOST')}:{os.getenv('PHOENIX_PORT', '6006')}/v1/traces",
        ),
        description="Arize Phoenix OTLP collector endpoint",
    )
    phoenix_project_name: str = Field(
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "story-rp-engine"),
        description="Arize Phoenix project name",
    )
    compaction_enabled: bool = Field(
        default_factory=lambda: os.getenv("STORY_RP_COMPACTION_ENABLED", "1").lower() in ("1", "true", "yes"),
        description="Whether ADK context compaction is enabled for agent and workflow sessions",
    )
    compaction_token_threshold: Optional[int] = Field(
        default_factory=lambda: int(os.getenv("STORY_RP_COMPACTION_TOKEN_THRESHOLD", "4000"))
        if os.getenv("STORY_RP_COMPACTION_TOKEN_THRESHOLD") not in (None, "")
        else 4000,
        description="Token limit that triggers compaction before model invocations",
    )
    compaction_event_retention_size: Optional[int] = Field(
        default_factory=lambda: int(os.getenv("STORY_RP_COMPACTION_EVENT_RETENTION_SIZE", "5"))
        if os.getenv("STORY_RP_COMPACTION_EVENT_RETENTION_SIZE") not in (None, "")
        else 5,
        description="Number of recent raw events kept intact when token threshold compaction occurs",
    )
    compaction_interval: Optional[int] = Field(
        default_factory=lambda: int(os.getenv("STORY_RP_COMPACTION_INTERVAL", "10"))
        if os.getenv("STORY_RP_COMPACTION_INTERVAL") not in (None, "")
        else 10,
        description="Number of turns between sliding window compactions",
    )
    compaction_overlap_size: Optional[int] = Field(
        default_factory=lambda: int(os.getenv("STORY_RP_COMPACTION_OVERLAP_SIZE", "2"))
        if os.getenv("STORY_RP_COMPACTION_OVERLAP_SIZE") not in (None, "")
        else 2,
        description="Number of prior turns to retain as overlapping context in sliding window compactions",
    )
    compaction_prompt_template: Optional[str] = Field(
        default_factory=lambda: os.getenv("STORY_RP_COMPACTION_PROMPT_TEMPLATE") or None,
        description="Optional custom summarizer prompt template containing {conversation_history}",
    )
