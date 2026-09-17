from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from story_rp_engine.api.routes_rp import router as rp_router
from story_rp_engine.api.routes_story import router as story_router
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore


def create_app(
    store: Optional[EngineStore] = None,
    config: Optional[EngineConfig] = None,
) -> FastAPI:
    app = FastAPI(title="Dual-Mode Story & Roleplay Engine API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.store = store or EngineStore()
    app.state.config = config or EngineConfig()

    app.include_router(rp_router)
    app.include_router(story_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
