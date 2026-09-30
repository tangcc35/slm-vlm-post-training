import os
from unittest.mock import patch
import pytest
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from google.adk.models.lite_llm import LiteLlm


@pytest.fixture(autouse=True)
def clean_environ(monkeypatch):
    """Ensure os.environ changes made during tests do not leak across tests."""
    orig_env = dict(os.environ)
    yield
    for k in list(os.environ.keys()):
        if k not in orig_env:
            monkeypatch.delenv(k, raising=False)
        elif os.environ[k] != orig_env[k]:
            monkeypatch.setenv(k, orig_env[k])


def test_get_adk_model_ollama(monkeypatch):
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = EngineConfig(model_name="ollama/llama3.1:8b", api_base="http://localhost:11434")
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert getattr(model, "model", None) == "ollama/llama3.1:8b" or getattr(model, "model_name", None) == "ollama/llama3.1:8b"
    assert os.environ.get("LITELLM_API_BASE") == "http://localhost:11434"


def test_get_adk_model_openai_compat(monkeypatch):
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = EngineConfig(
        model_name="openai/gemma-2-9b-it",
        api_base="http://localhost:8000/v1",
        api_key="custom-key",
    )
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert getattr(model, "model", None) == "openai/gemma-2-9b-it" or getattr(model, "model_name", None) == "openai/gemma-2-9b-it"
    assert os.environ.get("OPENAI_API_KEY") == "custom-key"


def test_get_adk_model_openai_compat_defaults_local_key(monkeypatch):
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = EngineConfig(
        model_name="openai/qwen3.5-2b",
        api_base="http://localhost:8001/v1",
    )
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert getattr(model, "model", None) == "openai/qwen3.5-2b" or getattr(model, "model_name", None) == "openai/qwen3.5-2b"
    assert os.environ.get("OPENAI_API_KEY") == "local"


def test_get_adk_model_defaults(monkeypatch):
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = EngineConfig(model_name="ollama/phi3:mini")
    model = get_adk_model(config)
    assert isinstance(model, LiteLlm)
    assert getattr(model, "model", None) == "ollama/phi3:mini" or getattr(model, "model_name", None) == "ollama/phi3:mini"
    assert "LITELLM_API_BASE" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_get_adk_model_fallback_handling(monkeypatch):
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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


def test_get_adk_model_gemini_native(monkeypatch):
    from google.adk.models.google_llm import Gemini
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)

    config = EngineConfig(model_name="gemini-flash-latest", api_key="test-gemini-key")
    model = get_adk_model(config)

    assert isinstance(model, Gemini)
    assert model.model == "gemini-flash-latest"
    assert model.retry_options is not None
    assert model.retry_options.attempts == 2
    assert os.environ.get("GOOGLE_API_KEY") == "test-gemini-key"


def test_get_adk_model_gemini_prefix_normalization(monkeypatch):
    from google.adk.models.google_llm import Gemini
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config = EngineConfig(model_name="gemini/gemini-2.5-flash")
    model = get_adk_model(config)

    assert isinstance(model, Gemini)
    assert model.model == "gemini-2.5-flash"


def test_get_adk_model_gemini_google_prefix_normalization(monkeypatch):
    from google.adk.models.google_llm import Gemini
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config = EngineConfig(model_name="google/gemini-2.0-flash")
    model = get_adk_model(config)

    assert isinstance(model, Gemini)
    assert model.model == "gemini-2.0-flash"


def test_engine_config_api_key_resolution_google_and_gemini_keys(monkeypatch):
    monkeypatch.delenv("STORY_RP_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    monkeypatch.setenv("GOOGLE_API_KEY", "google-key-123")
    cfg1 = EngineConfig()
    assert cfg1.api_key == "google-key-123"

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-456")
    cfg2 = EngineConfig()
    assert cfg2.api_key == "gemini-key-456"


def test_is_gemini_model():
    from story_rp_engine.core.model_provider import is_gemini_model
    assert is_gemini_model("gemini-flash-latest")
    assert is_gemini_model("gemini-2.5-flash")
    assert is_gemini_model("gemini/gemini-2.5-pro")
    assert is_gemini_model("google/gemini-2.0-flash")
    assert not is_gemini_model("ollama/llama3.1:8b")
    assert not is_gemini_model("openai/qwen3.5-2b")


def test_is_remote_model():
    from story_rp_engine.core.model_provider import is_remote_model
    # Gemini models are remote
    assert is_remote_model(EngineConfig(model_name="gemini-flash-latest"))
    assert is_remote_model(EngineConfig(model_name="google/gemini-2.5-flash"))
    # Cloud providers are remote
    assert is_remote_model(EngineConfig(model_name="anthropic/claude-3-5-sonnet"))
    assert is_remote_model(EngineConfig(model_name="vertex_ai/gemini-1.5-flash"))
    assert is_remote_model(EngineConfig(model_name="openai/gpt-4o"))
    # Local providers are not remote
    assert not is_remote_model(EngineConfig(model_name="ollama/llama3.1:8b"))
    assert not is_remote_model(EngineConfig(model_name="openai/qwen3.5-2b", api_base="http://127.0.0.1:8001/v1"))
    assert not is_remote_model(EngineConfig(model_name="openai/custom", api_base="http://localhost:8080/v1"))



