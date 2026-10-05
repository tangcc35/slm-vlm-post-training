import inspect
import asyncpg
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.db import engine_kwargs_for, normalize_db_url, redact_db_url

# The format Neon's console and Vercel integration hand out.
NEON_URL = (
    "postgresql://alex:AbC123dEf@ep-cool-darkness-123456-pooler.us-east-2.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)


def test_normalize_db_url_empty():
    assert normalize_db_url(None) is None
    assert normalize_db_url("") is None
    assert normalize_db_url("   ") is None


def test_normalize_neon_url_for_asyncpg():
    url = make_url(normalize_db_url(NEON_URL))
    assert url.drivername == "postgresql+asyncpg"
    assert dict(url.query) == {"ssl": "require"}
    assert url.username == "alex"
    assert url.password == "AbC123dEf"
    assert url.host == "ep-cool-darkness-123456-pooler.us-east-2.aws.neon.tech"
    assert url.database == "neondb"


def test_normalized_neon_url_only_passes_args_asyncpg_accepts():
    # SQLAlchemy forwards URL query params to asyncpg.connect() as keyword arguments;
    # channel_binding used to crash here with "unexpected keyword argument".
    engine = create_async_engine(normalize_db_url(NEON_URL))
    _, connect_kwargs = engine.dialect.create_connect_args(engine.url)
    assert set(connect_kwargs) <= set(inspect.signature(asyncpg.connect).parameters)


def test_normalize_postgres_scheme_alias():
    url = make_url(normalize_db_url("postgres://u:p@db.example.com:5433/app"))
    assert url.drivername == "postgresql+asyncpg"
    assert url.port == 5433
    assert dict(url.query) == {}


def test_normalize_neon_host_defaults_to_ssl():
    url = make_url(normalize_db_url("postgresql://u:p@ep-x.us-east-2.aws.neon.tech/neondb"))
    assert url.query["ssl"] == "require"


def test_normalize_explicit_ssl_wins_over_sslmode():
    url = make_url(normalize_db_url("postgresql://u:p@ep-x.aws.neon.tech/db?ssl=verify-full&sslmode=require"))
    assert dict(url.query) == {"ssl": "verify-full"}


def test_normalize_drops_libpq_only_params():
    url = make_url(
        normalize_db_url(
            "postgresql://u:p@ep-x-pooler.aws.neon.tech/db"
            "?sslmode=require&pgbouncer=true&connect_timeout=15&options=endpoint%3Dep-x&application_name=app"
        )
    )
    assert dict(url.query) == {"ssl": "require"}


def test_normalize_keeps_unix_socket_host():
    url = make_url(normalize_db_url("postgresql://postgres:@/postgres?host=/run/pg"))
    assert url.drivername == "postgresql+asyncpg"
    assert dict(url.query) == {"host": "/run/pg"}


def test_normalize_preserves_special_characters_in_password():
    url = make_url(normalize_db_url("postgresql://u:p%40ss%3Aw%2Frd%23@ep-x.aws.neon.tech/db?sslmode=require"))
    assert url.password == "p@ss:w/rd#"


def test_normalize_cleans_explicit_asyncpg_url():
    url = make_url(normalize_db_url(NEON_URL.replace("postgresql://", "postgresql+asyncpg://")))
    assert dict(url.query) == {"ssl": "require"}


def test_normalize_leaves_other_drivers_alone():
    psycopg_url = NEON_URL.replace("postgresql://", "postgresql+psycopg://")
    assert make_url(normalize_db_url(psycopg_url)) == make_url(psycopg_url)


def test_normalize_sqlite():
    assert normalize_db_url("sqlite:///custom.db") == "sqlite+aiosqlite:///custom.db"
    assert normalize_db_url("sqlite+aiosqlite:///custom.db") == "sqlite+aiosqlite:///custom.db"


def test_redact_db_url_hides_password():
    redacted = redact_db_url(normalize_db_url(NEON_URL))
    assert "AbC123dEf" not in redacted
    assert "ep-cool-darkness-123456-pooler" in redacted


def test_engine_kwargs_null_pool_only_for_server_databases():
    pg_url = normalize_db_url(NEON_URL)
    assert engine_kwargs_for(pg_url, null_pool=True) == {"poolclass": NullPool}
    assert engine_kwargs_for(pg_url, null_pool=False) == {}
    assert engine_kwargs_for("sqlite+aiosqlite:///x.db", null_pool=True) == {}


def test_config_db_url_precedence(monkeypatch):
    for name in ("STORY_RP_DB_URL", "DATABASE_URL", "POSTGRES_URL"):
        monkeypatch.delenv(name, raising=False)
    assert EngineConfig().db_url is None

    monkeypatch.setenv("POSTGRES_URL", "postgresql://pg_user:p@host/db")
    assert "pg_user" in EngineConfig().db_url

    monkeypatch.setenv("DATABASE_URL", "postgresql://db_user:p@host/db")
    assert "db_user" in EngineConfig().db_url

    monkeypatch.setenv("STORY_RP_DB_URL", "postgresql://story_user:p@host/db")
    assert "story_user" in EngineConfig().db_url


def test_config_db_null_pool(monkeypatch):
    monkeypatch.delenv("STORY_RP_DB_NULL_POOL", raising=False)
    assert EngineConfig().db_null_pool is False
    monkeypatch.setenv("STORY_RP_DB_NULL_POOL", "1")
    assert EngineConfig().db_null_pool is True
