from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.ingestion import IngestionError, IngestionService, get_ingestion_service

router = APIRouter()
logger = get_logger("app.api.admin")

_last_refresh: datetime | None = None


class IngestTextRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier")
    source: str = Field(..., description="Source label")
    texts: list[str] = Field(..., description="Raw texts or markdown bodies")


class IngestResponse(BaseModel):
    document_id: str
    chunks: int


@router.post("/ingest/text", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_text(
    payload: IngestTextRequest, svc: IngestionService = Depends(get_ingestion_service)
) -> IngestResponse:
    try:
        result = await svc.ingest_texts(tenant_id=payload.tenant_id, source=payload.source, texts=payload.texts)
    except IngestionError as exc:
        logger.exception("admin.ingest_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc))
    global _last_refresh
    _last_refresh = datetime.now(timezone.utc)
    return IngestResponse(document_id=result.document_id, chunks=result.chunks_count)


@router.get("/status")
async def ingestion_status() -> dict[str, Any]:
    status_str = _last_refresh.isoformat() if _last_refresh else "never"
    return {"last_refresh": status_str}
