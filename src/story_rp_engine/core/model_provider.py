"""Model provider factory for Google ADK models (Gemini & LiteLLM)."""

import os
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.genai import types
from story_rp_engine.core.config import EngineConfig


def is_gemini_model(model_name: str) -> bool:
    """Checks whether the specified model identifier represents a Google Gemini model."""
    clean_model = model_name.strip()
    for prefix in ("gemini/", "google/"):
        if clean_model.startswith(prefix):
            clean_model = clean_model[len(prefix):]
            break
    return clean_model.startswith("gemini")


def is_remote_model(config: EngineConfig) -> bool:
    """Determines if the model runs via a remote cloud API where local server warmup is unnecessary."""
    if is_gemini_model(config.model_name):
        return True
    model = config.model_name.lower()
    if model.startswith(("anthropic/", "vertex_ai/", "groq/", "azure/")):
        return True
    if model.startswith("openai/"):
        if not config.api_base or not ("localhost" in config.api_base or "127.0.0.1" in config.api_base):
            return True
        return False
    if model.startswith("ollama/"):
        return False
    return False


def get_generate_config(config: EngineConfig) -> types.GenerateContentConfig:
    """Sampling settings from the engine config, applied to every agent."""
    return types.GenerateContentConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_tokens,
    )


def get_adk_model(config: EngineConfig) -> BaseLlm:
    """Instantiates and configures a Google ADK model wrapper (Gemini or LiteLlm)."""
    api_key = config.api_key
    model_name = config.model_name

    # Determine if this is a Gemini model
    clean_model = model_name
    for prefix in ("gemini/", "google/"):
        if clean_model.startswith(prefix):
            clean_model = clean_model[len(prefix):]
            break

    is_gemini = is_gemini_model(model_name)

    if is_gemini:
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        gemini_kwargs = {
            "retry_options": types.HttpRetryOptions(initial_delay=1, attempts=2),
        }
        if api_key:
            gemini_kwargs["client_kwargs"] = {"api_key": api_key}
        if config.api_base and not ("127.0.0.1" in config.api_base or "localhost" in config.api_base):
            gemini_kwargs["base_url"] = config.api_base

        return Gemini(model=clean_model, **gemini_kwargs)

    # For other models, fallback to LiteLlm
    from google.adk.models.lite_llm import LiteLlm

    if not api_key and config.api_base and model_name.startswith("openai/"):
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
        return LiteLlm(model_name=model_name, **kwargs)
    except TypeError:
        # Fallback for alternative parameter signature
        return LiteLlm(model=model_name, **kwargs)

