from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.db.models.document import DocumentChunk
from app.db.models.structured_data import CustomerAccount
from app.services.embeddings import EmbeddingProvider, OpenAIEmbeddingClient

logger = get_logger("app.retrieval")


class RetrievalError(Exception):
    """Raised when retrieval fails."""


@dataclass
class RetrievedContext:
    id: str
    score: float
    content: str
    source: str


class RetrievalService:
    def __init__(
        self,
        session_factory: sessionmaker[AsyncSession],
        settings: AppSettings | None = None,
        embeddings: EmbeddingProvider | None = None,
    ):
        self._session_factory = session_factory
        self._settings = settings or get_settings()
        self._embeddings = embeddings or OpenAIEmbeddingClient(self._settings)

    async def retrieve(self, question: str, tenant_id: str | None) -> list[RetrievedContext]:
        if tenant_id is None:
            raise RetrievalError("tenant_id is required for retrieval")

        start = time.perf_counter()
        query_embedding = await self._embed_query(question)
        async with self._session_factory() as session:
            docs = await self._fetch_candidate_chunks(session, tenant_id)
            structured = await self._fetch_structured_data(session, tenant_id)

        scored = self._score_chunks(query_embedding, docs)
        merged = scored + structured
        merged.sort(key=lambda ctx: ctx.score, reverse=True)
        cutoff = self._settings.similarity_threshold
        limited = [ctx for ctx in merged if ctx.score >= cutoff][: self._settings.max_context_chunks]

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "retrieval.complete",
            extra={"latency_ms": latency_ms, "tenant_id": tenant_id, "results": len(limited)},
        )
        return limited

    async def _embed_query(self, question: str) -> list[float]:
        embeddings = await self._embeddings.embed_texts([question])
        if not embeddings:
            raise RetrievalError("empty embedding response")
        return embeddings[0]

    async def _fetch_candidate_chunks(self, session: AsyncSession, tenant_id: str) -> list[DocumentChunk]:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.tenant_id == tenant_id)
            .order_by(DocumentChunk.created_at.desc())
            .limit(200)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _fetch_structured_data(self, session: AsyncSession, tenant_id: str) -> list[RetrievedContext]:
        stmt = select(CustomerAccount).where(CustomerAccount.tenant_id == tenant_id).limit(20)
        result = await session.execute(stmt)
        accounts = list(result.scalars().all())
        contexts: list[RetrievedContext] = []
        for account in accounts:
            contexts.append(
                RetrievedContext(
                    id=str(account.id),
                    score=0.5,  # baseline so chunks usually win; adjusted in tests
                    content=account.summary(),
                    source="customer_account",
                )
            )
        return contexts

    def _score_chunks(
        self, query_embedding: Sequence[float], chunks: Iterable[DocumentChunk]
    ) -> list[RetrievedContext]:
        scored: list[RetrievedContext] = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            score = self._cosine_similarity(query_embedding, chunk.embedding)
            scored.append(chunk.as_context(score=score))  # type: ignore[arg-type]
        return [RetrievedContext(**ctx) if not isinstance(ctx, RetrievedContext) else ctx for ctx in scored]

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def get_retrieval_service(
    session_factory: sessionmaker[AsyncSession], settings: AppSettings | None = None
) -> RetrievalService:
    return RetrievalService(session_factory=session_factory, settings=settings)
