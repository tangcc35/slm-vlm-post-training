import os
from unittest.mock import patch
import pytest
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from google.adk.models.lite_llm import LiteLlm


def test_get_adk_model_ollama():
    config = EngineConfig(model_name="ollama/llama3.1:8b", api_base="http://localhost:11434")
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert getattr(model, "model", None) == "ollama/llama3.1:8b" or getattr(model, "model_name", None) == "ollama/llama3.1:8b"
    assert os.environ.get("LITELLM_API_BASE") == "http://localhost:11434"


def test_get_adk_model_openai_compat():
    config = EngineConfig(
        model_name="openai/gemma-2-9b-it",
        api_base="http://localhost:8000/v1",
        api_key="custom-key",
    )
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert getattr(model, "model", None) == "openai/gemma-2-9b-it" or getattr(model, "model_name", None) == "openai/gemma-2-9b-it"
    assert os.environ.get("OPENAI_API_KEY") == "custom-key"


def test_get_adk_model_defaults():
    config = EngineConfig(model_name="ollama/phi3:mini")
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert getattr(model, "model", None) == "ollama/phi3:mini" or getattr(model, "model_name", None) == "ollama/phi3:mini"


def test_get_adk_model_fallback_handling():
    config = EngineConfig(model_name="ollama/test-model")

    # Simulate LiteLlm where model_name argument raises TypeError, falling back to model argument
    call_records = []

    def mock_lite_llm(*args, **kwargs):
        call_records.append(kwargs)
        if "model_name" in kwargs:
            raise TypeError("unexpected keyword argument 'model_name'")
        # Return real LiteLlm instance
        return LiteLlm(model=kwargs.get("model", "ollama/test-model"))

    with patch("story_rp_engine.core.model_provider.LiteLlm", side_effect=mock_lite_llm):
        model = get_adk_model(config)
        assert isinstance(model, LiteLlm)
        assert len(call_records) == 2
        assert "model_name" in call_records[0]
        assert "model" in call_records[1]
