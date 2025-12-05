from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import get_engine, get_session_factory


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title="LLM Knowledge Assistant", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        logging.getLogger("app.lifecycle").info("app.startup", extra={"env": settings.env})
        # Ensures engine/session factory initialized
        get_engine()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logging.getLogger("app.lifecycle").info("app.shutdown")
        engine = get_engine()
        await engine.dispose()

    app.include_router(health_router, tags=["ops"])
    app.include_router(chat_router, tags=["chat"])
    app.include_router(admin_router, prefix="/admin", tags=["admin"])

    return app


app = create_app()
