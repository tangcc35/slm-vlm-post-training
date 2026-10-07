import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from story_rp_engine.api.routes_group import router as group_router
from story_rp_engine.api.routes_lorebook import router as lorebook_router
from story_rp_engine.api.routes_persona import router as persona_router
from story_rp_engine.api.routes_rp import router as rp_router
from story_rp_engine.api.routes_story import router as story_router
from story_rp_engine.core.agent_registry import AgentRegistry
from story_rp_engine.core.agent_utils import execute_runner_turn
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import is_gemini_model, is_remote_model
from story_rp_engine.storage.store import EngineStore

logger = logging.getLogger("story_rp_engine.api")


class RevalidatingStaticFiles(StaticFiles):
    """Makes browsers check with the server before reusing a cached UI file, so frontend updates show up."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    skip_warmup = os.getenv("STORY_RP_SKIP_WARMUP", "0").lower() in ("1", "true")
    config = getattr(app.state, "config", None)
    if config is not None and is_remote_model(config):
        skip_warmup = True
        logger.info("Skipping model warmup for remote model (%s).", config.model_name)
    elif config is None:
        env_model = os.getenv("STORY_RP_MODEL", "")
        if env_model and is_gemini_model(env_model):
            skip_warmup = True
            logger.info("Skipping model warmup for remote model (%s).", env_model)

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

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        missing_fields = []
        seen_missing = set()
        other_errors = []

        for err in exc.errors():
            field_name = str(err["loc"][-1]) if err.get("loc") else "unknown"
            if err.get("type") == "missing":
                if field_name not in seen_missing:
                    seen_missing.add(field_name)
                    missing_fields.append(field_name)
            else:
                other_errors.append(f"{field_name}: {err.get('msg')}")

        messages = []
        if missing_fields:
            messages.append(f"Missing required fields: {', '.join(missing_fields)}")
        if other_errors:
            messages.append(f"Invalid fields: {'; '.join(other_errors)}")

        error_msg = " | ".join(messages) if messages else "Invalid request payload"

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": error_msg,
                "detail": error_msg,
                "missing_fields": missing_fields,
            },
        )

    resolved_config = config or EngineConfig()
    if resolved_config.phoenix_enabled:
        try:
            from phoenix.otel import register
            register(
                project_name=resolved_config.phoenix_project_name,
                endpoint=resolved_config.phoenix_endpoint,
                auto_instrument=True,
            )
        except ImportError:
            logger.warning("PHOENIX_ENABLED is true, but phoenix is not installed. Skipping OpenTelemetry registration.")

    resolved_store = store or EngineStore(
        storage_dir=resolved_config.storage_dir,
        db_url=resolved_config.db_url,
        db_null_pool=resolved_config.db_null_pool,
    )

    app.state.store = resolved_store
    app.state.config = resolved_config
    app.state.session_service = resolved_store.session_service
    app.state.agent_registry = AgentRegistry(resolved_config, resolved_store)
    app.state.runner = app.state.agent_registry.get_story_runner()

    app.include_router(rp_router)
    app.include_router(story_router)
    app.include_router(lorebook_router)
    app.include_router(group_router)
    app.include_router(persona_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        app.mount("/", RevalidatingStaticFiles(directory=str(web_dir), html=True), name="web_ui")

    return app
