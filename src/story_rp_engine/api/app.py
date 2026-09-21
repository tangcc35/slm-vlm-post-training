import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from story_rp_engine.api.routes_rp import router as rp_router
from story_rp_engine.api.routes_story import router as story_router
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.agent_utils import execute_runner_turn
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore

logger = logging.getLogger("story_rp_engine.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    skip_warmup = os.getenv("STORY_RP_SKIP_WARMUP", "0").lower() in ("1", "true")
    runner = getattr(app.state, "runner", None)
    if runner is not None and not skip_warmup:
        try:
            logger.info("Performing model warmup on startup...")
            await execute_runner_turn(
                runner=runner,
                user_id="SystemWarmup",
                session_id="warmup_session",
                message="Hi",
                state_delta={
                    "premise": "Warmup",
                    "genre": "General",
                    "tone": "Neutral",
                    "current_text": "",
                    "instruction": "Say ready.",
                },
            )
            logger.info("Model warmup completed successfully.")
        except Exception as e:
            logger.warning("Startup model warmup skipped: %s", e)
    yield


def create_app(
    store: Optional[EngineStore] = None,
    config: Optional[EngineConfig] = None,
) -> FastAPI:
    app = FastAPI(
        title="Dual-Mode Story & Roleplay Engine API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolved_config = config or EngineConfig()
    resolved_store = store or EngineStore(
        storage_dir=resolved_config.storage_dir,
        db_url=resolved_config.db_url,
    )

    app.state.store = resolved_store
    app.state.config = resolved_config
    app.state.session_service = resolved_store.session_service
    app.state.agent_registry = AgentRegistry(resolved_config, resolved_store)
    app.state.runner = app.state.agent_registry.get_story_runner()

    app.include_router(rp_router)
    app.include_router(story_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
