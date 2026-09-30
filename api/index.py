import os
import sys
from pathlib import Path

# Add project root and src to sys.path so story_rp_engine can be resolved
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# Serverless environment defaults
os.environ.setdefault("STORY_RP_STORAGE_DIR", "/tmp/.engine_data")
os.environ.setdefault("STORY_RP_SKIP_WARMUP", "1")
os.environ.setdefault("PHOENIX_ENABLED", "0")

from story_rp_engine.api.app import create_app

# Vercel looks for the WSGI/ASGI application callable named 'app'
app = create_app()
