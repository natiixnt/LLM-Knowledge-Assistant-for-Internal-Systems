from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import AppSettings, DatabaseSettings, OpenAISettings
from app.db.base import Base
from app.db.models.document import Document, DocumentChunk
from app.services.retrieval import RetrievalService


class FakeEmbeddings:
    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_retrieval_orders_by_similarity():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    doc = Document(tenant_id="t1", title="Doc", source="s1")
    chunk_good = DocumentChunk(
        tenant_id="t1",
        document_id=doc.id,
        content="target",
        embedding=[1.0, 0.0],
        position=0,
        chunk_metadata={"source": "good"},
        document=doc,
    )
    chunk_bad = DocumentChunk(
        tenant_id="t1",
        document_id=doc.id,
        content="other",
        embedding=[0.0, 1.0],
        position=1,
        chunk_metadata={"source": "bad"},
        document=doc,
    )

    async with async_session() as session:
        async with session.begin():
            session.add_all([doc, chunk_good, chunk_bad])

    settings = AppSettings(
        env="test",
        db=DatabaseSettings(url="sqlite+aiosqlite:///:memory:", echo=False),
        openai=OpenAISettings(api_key="test-key", model="gpt-3.5-turbo", embedding_model="text-embedding-3-small"),
        similarity_threshold=0.0,
        max_context_chunks=2,
    )

    svc = RetrievalService(session_factory=async_session, settings=settings, embeddings=FakeEmbeddings())
    contexts = await svc.retrieve(question="hi", tenant_id="t1")

    assert contexts
    assert contexts[0].source == "good"
    assert contexts[0].score >= contexts[1].score
