import os
import sys
from pathlib import Path

# Resolve directory paths for local, root, and subfolder execution
CURRENT_DIR = Path(__file__).resolve().parent  # vercel/api
VERCEL_DIR = CURRENT_DIR.parent                # vercel
REPO_ROOT = VERCEL_DIR.parent                 # repo root

for candidate in (
    VERCEL_DIR / "src",
    VERCEL_DIR,
    REPO_ROOT / "src",
    REPO_ROOT,
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Serverless environment defaults
os.environ.setdefault("STORY_RP_STORAGE_DIR", "/tmp/.engine_data")
os.environ.setdefault("STORY_RP_SKIP_WARMUP", "1")
os.environ.setdefault("STORY_RP_DB_NULL_POOL", "1")
os.environ["PHOENIX_ENABLED"] = "0"

import logging

from story_rp_engine.api.app import create_app

base_app = create_app()
if base_app.state.config.db_url is None:
    logging.getLogger("story_rp_engine.vercel").warning(
        "No DATABASE_URL configured: characters, lorebooks, groups, personas and sessions are kept in /tmp "
        "and lost whenever this instance is recycled."
    )


class VercelPathMiddleware:
    """Restores the original request path if Vercel internal rewrites rewrite the path to the entrypoint filename."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in ("/api/index.py", "/api/index", "/api"):
                headers = dict(scope.get("headers", []))
                raw_matched = headers.get(b"x-matched-path", b"").decode("utf-8")
                raw_forwarded = headers.get(b"x-forwarded-uri", b"").decode("utf-8")
                raw_vercel_path = headers.get(b"x-vercel-matched-path", b"").decode("utf-8")
                orig_path = raw_matched or raw_forwarded or raw_vercel_path or "/"
                if orig_path not in ("/api/index.py", "/api/index", "/api"):
                    scope["path"] = orig_path.split("?")[0]
                else:
                    scope["path"] = "/"
        await self.app(scope, receive, send)


app = VercelPathMiddleware(base_app)
