import logging
from typing import Any, Dict, Optional
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

logger = logging.getLogger("story_rp_engine.storage")

# SQLAlchemy's asyncpg dialect passes every URL query parameter straight to
# asyncpg.connect() as a keyword argument, so libpq-only parameters found in
# Neon/Vercel connection strings (channel_binding, connect_timeout, options, ...)
# would raise TypeError on connect. Only these string-valued ones are kept.
_ASYNCPG_QUERY_PARAMS = {
    "ssl",
    "host",
    "port",
    "target_session_attrs",
    "prepared_statement_cache_size",
    "passfile",
    "service",
    "servicefile",
    "krbsrvname",
    "gsslib",
}


def normalize_db_url(url: Optional[str]) -> Optional[str]:
    """Converts a database URL (e.g. a Neon connection string) into an async SQLAlchemy URL."""
    if not url or not url.strip():
        return None

    parsed = make_url(url.strip())
    backend = parsed.get_backend_name()

    if backend == "sqlite":
        if parsed.drivername == "sqlite":
            parsed = parsed.set(drivername="sqlite+aiosqlite")
        return parsed.render_as_string(hide_password=False)

    if backend not in ("postgres", "postgresql"):
        return parsed.render_as_string(hide_password=False)

    driver = parsed.drivername.partition("+")[2]
    if driver and driver != "asyncpg":
        # An explicitly chosen driver (e.g. psycopg) understands libpq parameters.
        return parsed.render_as_string(hide_password=False)

    query = dict(parsed.query)
    sslmode = query.pop("sslmode", None)
    if sslmode and "ssl" not in query:
        query["ssl"] = sslmode
    if "ssl" not in query and (parsed.host or "").endswith(".neon.tech"):
        query["ssl"] = "require"

    dropped = sorted(k for k in query if k not in _ASYNCPG_QUERY_PARAMS)
    if dropped:
        logger.info("Ignoring database URL parameters unsupported by asyncpg: %s", ", ".join(dropped))
    query = {k: v for k, v in query.items() if k in _ASYNCPG_QUERY_PARAMS}

    parsed = parsed.set(drivername="postgresql+asyncpg", query=query)
    return parsed.render_as_string(hide_password=False)


def redact_db_url(url: str) -> str:
    """Returns the URL with its password masked, for logging."""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return "<unparseable database URL>"


def engine_kwargs_for(url: str, null_pool: bool) -> Dict[str, Any]:
    """Extra create_async_engine() arguments for a normalized database URL."""
    if null_pool and make_url(url).get_backend_name() != "sqlite":
        # Serverless instances can be frozen or recycled between requests, leaving
        # pooled connections dead; Neon's pooled (-pooler) endpoint already pools
        # server-side and Neon advises against client-side pooling on top of it.
        return {"poolclass": NullPool}
    return {}
