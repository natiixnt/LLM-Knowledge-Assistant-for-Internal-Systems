from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.db.models.document import Document, DocumentChunk
from app.db.models.structured_data import CustomerAccount
from app.services.embeddings import EmbeddingProvider, OpenAIEmbeddingClient
from app.db.session import get_session_factory
from fastapi import Depends
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger("app.ingestion")


class IngestionError(Exception):
    pass


@dataclass
class IngestionResult:
    document_id: str
    chunks_count: int


class IngestionService:
    def __init__(
        self,
        session_factory: sessionmaker[AsyncSession],
        embeddings: EmbeddingProvider | None = None,
        settings: AppSettings | None = None,
    ):
        self._session_factory = session_factory
        self._settings = settings or get_settings()
        self._embeddings = embeddings or OpenAIEmbeddingClient(self._settings)

    async def ingest_texts(self, *, tenant_id: str, source: str, texts: Sequence[str]) -> IngestionResult:
        if not texts:
            raise IngestionError("No texts provided")

        chunks = list(self._chunk_texts(texts))
        embeddings = await self._embeddings.embed_texts([c["content"] for c in chunks])
        if len(embeddings) != len(chunks):
            raise IngestionError("Embedding count mismatch")

        async with self._session_factory() as session:
            async with session.begin():
                document = Document(tenant_id=tenant_id, title=f"Ingested from {source}", source=source)
                document.chunks = []
                session.add(document)
                await session.flush()

                for idx, chunk in enumerate(chunks):
                    document.chunks.append(
                        DocumentChunk(
                            tenant_id=tenant_id,
                            document_id=document.id,
                            content=chunk["content"],
                            embedding=embeddings[idx],
                            position=chunk["position"],
                            chunk_metadata={"source": source, "chunk": idx},
                        )
                    )

            await session.commit()
            logger.info(
                "ingestion.document_stored", extra={"document_id": str(document.id), "chunks": len(chunks)}
            )
            return IngestionResult(document_id=str(document.id), chunks_count=len(chunks))

    async def upsert_customer_accounts(self, *, tenant_id: str, accounts: Iterable[dict]) -> int:
        accounts_list = list(accounts)
        async with self._session_factory() as session:
            async with session.begin():
                for account in accounts_list:
                    session.add(
                        CustomerAccount(
                            tenant_id=tenant_id,
                            name=account["name"],
                            segment=account.get("segment", "unknown"),
                            arr=float(account.get("arr", 0)),
                            notes=account.get("notes", ""),
                        )
                    )
        logger.info(
            "ingestion.accounts_upserted", extra={"count": len(accounts_list), "tenant_id": tenant_id}
        )
        return len(accounts_list)

    def _chunk_texts(self, texts: Sequence[str]) -> Iterable[dict[str, float | str]]:
        max_chunk_size = 500
        position = 0
        for text in texts:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                for idx in range(0, len(para), max_chunk_size):
                    chunk = para[idx : idx + max_chunk_size]
                    yield {"content": chunk, "position": float(position)}
                    position += 1.0


def get_ingestion_service(
    session_factory: sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> IngestionService:
    settings = get_settings()
    return IngestionService(session_factory=session_factory, settings=settings)
