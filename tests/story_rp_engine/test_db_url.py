import pytest
from story_rp_engine.core.config import EngineConfig, normalize_db_url, resolve_db_url


def test_normalize_db_url_none():
    assert normalize_db_url(None) is None
    assert normalize_db_url("") is None


def test_normalize_db_url_postgres_scheme():
    url = "postgres://user:pass@ep-cool.us-east-2.aws.neon.tech/neondb"
    normalized = normalize_db_url(url)
    assert normalized.startswith("postgresql+asyncpg://")
    assert "ssl=require" in normalized


def test_normalize_db_url_postgresql_scheme():
    url = "postgresql://user:pass@ep-cool.us-east-2.aws.neon.tech/neondb?sslmode=require"
    normalized = normalize_db_url(url)
    assert normalized.startswith("postgresql+asyncpg://")
    assert "sslmode=require" not in normalized
    assert "ssl=require" in normalized


def test_normalize_db_url_already_asyncpg():
    url = "postgresql+asyncpg://user:pass@ep-cool.us-east-2.aws.neon.tech/neondb?ssl=require"
    normalized = normalize_db_url(url)
    assert normalized == url


def test_normalize_db_url_sqlite():
    url = "sqlite:///custom.db"
    normalized = normalize_db_url(url)
    assert normalized == "sqlite+aiosqlite:///custom.db"


def test_resolve_db_url_precedence(monkeypatch):
    monkeypatch.delenv("STORY_RP_DB_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)

    # When none set
    assert resolve_db_url() is None

    # NEON_DATABASE_URL fallback
    monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://neon_user:p@host/db")
    assert "neon_user" in resolve_db_url()

    # POSTGRES_URL overrides NEON_DATABASE_URL
    monkeypatch.setenv("POSTGRES_URL", "postgresql://pg_user:p@host/db")
    assert "pg_user" in resolve_db_url()

    # DATABASE_URL overrides POSTGRES_URL
    monkeypatch.setenv("DATABASE_URL", "postgresql://db_user:p@host/db")
    assert "db_user" in resolve_db_url()

    # STORY_RP_DB_URL overrides DATABASE_URL
    monkeypatch.setenv("STORY_RP_DB_URL", "postgresql://story_user:p@host/db")
    assert "story_user" in resolve_db_url()


def test_engine_config_db_url_auto_resolution(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://neon:pass@ep-xyz.neon.tech/db?sslmode=require")
    monkeypatch.delenv("STORY_RP_DB_URL", raising=False)
    config = EngineConfig()
    assert config.db_url is not None
    assert config.db_url.startswith("postgresql+asyncpg://")
    assert "ssl=require" in config.db_url
