"""Model provider factory for Google ADK & LiteLLM integration."""

import os
from google.adk.models.lite_llm import LiteLlm
from story_rp_engine.core.config import EngineConfig


def get_adk_model(config: EngineConfig) -> LiteLlm:
    """Instantiates and configures a Google ADK LiteLlm model wrapper."""
    api_key = config.api_key
    if not api_key and config.api_base and config.model_name.startswith("openai/"):
        api_key = "local"

    if config.api_base:
        os.environ["LITELLM_API_BASE"] = config.api_base
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # LiteLlm accepts model_name or model depending on version, and passes through keyword options
    kwargs = {}
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if api_key:
        kwargs["api_key"] = api_key

    try:
        return LiteLlm(model_name=config.model_name, **kwargs)
    except TypeError:
        # Fallback for alternative parameter signature
        return LiteLlm(model=config.model_name, **kwargs)
