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
os.environ.setdefault("PHOENIX_ENABLED", "0")

from story_rp_engine.api.app import create_app

app = create_app()
