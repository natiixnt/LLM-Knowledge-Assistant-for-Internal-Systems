from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, constr

from app.core.logging import get_logger
from app.services.chat import ChatResult, ChatService, LLMError, RetrievalError, get_chat_service

logger = get_logger("app.api.chat")

router = APIRouter()


class ChatRequest(BaseModel):
    question: constr(min_length=3) = Field(..., description="User question")
    user_id: str = Field(..., min_length=3)
    tenant_id: str | None = None
    trace_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int


async def _with_timeout(coro, *, timeout: float):
    return await asyncio.wait_for(coro, timeout=timeout)


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, svc: ChatService = Depends(get_chat_service)) -> ChatResponse:
    ctx = {
        "trace_id": payload.trace_id or str(uuid4()),
        "user_id": payload.user_id,
        "tenant_id": payload.tenant_id,
    }

    logger.info("chat.request", extra=ctx | {"question": payload.question})

    try:
        result: ChatResult = await _with_timeout(
            svc.answer(question=payload.question, user_id=payload.user_id, tenant_id=payload.tenant_id),
            timeout=8.0,
        )
    except TimeoutError as exc:
        logger.warning("chat.timeout", extra=ctx)
        raise HTTPException(status_code=504, detail="Upstream LLM timeout") from exc
    except RetrievalError as exc:
        logger.exception("chat.retrieval_failed", extra=ctx | {"error": str(exc)})
        raise HTTPException(status_code=500, detail="Context retrieval failed") from exc
    except LLMError as exc:
        logger.exception("chat.llm_failed", extra=ctx | {"error": str(exc)})
        raise HTTPException(status_code=502, detail="LLM provider failed") from exc

    logger.info(
        "chat.success",
        extra=ctx | {"sources": result.sources, "latency_ms": result.latency_ms},
    )

    return ChatResponse(answer=result.answer, sources=result.sources, latency_ms=result.latency_ms)
