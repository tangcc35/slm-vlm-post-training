from typing import Optional
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    model_name: str = Field(default="ollama/llama3.1:8b", description="Model identifier for LiteLLM")
    api_base: Optional[str] = Field(default=None, description="Custom endpoint URL e.g. http://localhost:11434")
    api_key: Optional[str] = Field(default=None, description="API key if required (e.g. for Gemini/OpenAI)")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
