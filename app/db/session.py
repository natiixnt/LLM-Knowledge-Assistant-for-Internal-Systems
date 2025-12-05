from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import AppSettings, get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_engine(settings: AppSettings) -> AsyncEngine:
    global _engine, _session_factory
    if _engine:
        return _engine
    _engine = create_async_engine(settings.db.url, echo=settings.db.echo, future=True)
    _session_factory = async_sessionmaker(bind=_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine


def get_engine() -> AsyncEngine:
    settings = get_settings()
    return _ensure_engine(settings)


def get_session_factory() -> sessionmaker[AsyncSession]:
    settings = get_settings()
    _ensure_engine(settings)
    assert _session_factory is not None
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        yield session
