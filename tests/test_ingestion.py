from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import AppSettings, DatabaseSettings, OpenAISettings
from app.db.base import Base
from app.services.ingestion import IngestionService


class FakeEmbeddings:
    async def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


@pytest.mark.asyncio
async def test_ingest_texts_creates_chunks():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    settings = AppSettings(
        env="test",
        db=DatabaseSettings(url="sqlite+aiosqlite:///:memory:", echo=False),
        openai=OpenAISettings(api_key="dummy", model="gpt-3.5-turbo", embedding_model="text-embedding-3-small"),
    )

    svc = IngestionService(session_factory=async_session, embeddings=FakeEmbeddings(), settings=settings)
    result = await svc.ingest_texts(tenant_id="tenant", source="unit-test", texts=["para1\n\npara2"])

    assert result.document_id
    assert result.chunks_count == 2
