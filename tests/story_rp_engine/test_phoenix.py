import os
from unittest.mock import patch
import pytest
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.api.app import create_app


def test_engine_config_phoenix_defaults():
    config = EngineConfig()
    assert config.phoenix_enabled is False
    assert config.phoenix_endpoint == "http://127.0.0.1:6006/v1/traces"
    assert config.phoenix_project_name == "story-rp-engine"


def test_engine_config_phoenix_env_vars(monkeypatch):
    monkeypatch.setenv("PHOENIX_ENABLED", "1")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://custom-host:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "custom-project")

    config = EngineConfig()
    assert config.phoenix_enabled is True
    assert config.phoenix_endpoint == "http://custom-host:6006/v1/traces"
    assert config.phoenix_project_name == "custom-project"


def test_create_app_phoenix_disabled():
    config = EngineConfig(phoenix_enabled=False)
    with patch("phoenix.otel.register") as mock_register:
        app = create_app(config=config)
        assert app is not None
        mock_register.assert_not_called()


def test_create_app_phoenix_enabled():
    config = EngineConfig(
        phoenix_enabled=True,
        phoenix_endpoint="http://127.0.0.1:6006/v1/traces",
        phoenix_project_name="test-phoenix-project",
    )
    with patch("phoenix.otel.register") as mock_register:
        app = create_app(config=config)
        assert app is not None
        mock_register.assert_called_once_with(
            project_name="test-phoenix-project",
            endpoint="http://127.0.0.1:6006/v1/traces",
            auto_instrument=True,
        )


def test_create_app_phoenix_error_not_swallowed():
    config = EngineConfig(phoenix_enabled=True)
    with patch("phoenix.otel.register", side_effect=RuntimeError("Phoenix connection failed")):
        with pytest.raises(RuntimeError, match="Phoenix connection failed"):
            create_app(config=config)
