from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.db.base import Base


class EmbeddingType(TypeDecorator):
    """Stores embeddings as ARRAY on Postgres and JSON elsewhere."""

    impl = ARRAY(Float)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(ARRAY(Float))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Sequence[float] | None, dialect):
        if value is None:
            return value
        return list(map(float, value))

    def process_result_value(self, value, dialect):
        return value


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", lazy="selectin", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_chunks_tenant_doc", "tenant_id", "document_id"),
        Index("ix_chunks_vector", "tenant_id", "embedding", postgresql_using="ivfflat"),
    )

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    document_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingType, nullable=False)
    position: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="chunks", lazy="joined")

    def as_context(self, score: float) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "score": round(score, 4),
            "content": self.content,
            "source": self.metadata.get("source", self.document.source if self.document else "unknown"),
        }
