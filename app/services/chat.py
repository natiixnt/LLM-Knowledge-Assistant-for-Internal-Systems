from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from fastapi import Depends
from langchain_core.prompts import ChatPromptTemplate
from openai import AsyncOpenAI, OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.services.retrieval import RetrievalError, RetrievalService, RetrievedContext, get_retrieval_service

logger = get_logger("app.chat")


class LLMError(Exception):
    """Raised when the upstream LLM fails."""


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str: ...


class OpenAILLMClient:
    def __init__(self, settings: AppSettings):
        settings.require_openai()
        self._client = AsyncOpenAI(api_key=settings.openai.api_key, timeout=settings.openai.timeout_seconds)
        self._model = settings.openai.model
        self._timeout = settings.openai.timeout_seconds

    async def generate(self, prompt: str) -> str:
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an internal knowledge assistant. Always cite sources by id in answers.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                ),
                timeout=self._timeout,
            )
        except (TimeoutError, OpenAIError) as exc:  # pragma: no cover - network dependent
            raise LLMError(str(exc)) from exc
        choice = response.choices[0]
        return choice.message.content or ""


@dataclass
class ChatResult:
    answer: str
    sources: list[str]
    latency_ms: int
    retrieval_ms: int
    llm_ms: int


class ChatService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: LLMClient | None = None,
        settings: AppSettings | None = None,
    ):
        self._retrieval = retrieval_service
        self._llm = llm_client or OpenAILLMClient(settings or get_settings())
        self._settings = settings or get_settings()

    async def answer(self, question: str, user_id: str, tenant_id: str | None) -> ChatResult:
        ctx = {"user_id": user_id, "tenant_id": tenant_id}
        start = time.perf_counter()
        try:
            retrieval_start = time.perf_counter()
            contexts = await self._retrieval.retrieve(question=question, tenant_id=tenant_id)
            retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)
        except RetrievalError as exc:
            logger.exception("chat.retrieval_error", extra=ctx | {"error": str(exc)})
            raise

        prompt = self._build_prompt(question, contexts)

        llm_start = time.perf_counter()
        answer = await self._llm.generate(prompt)
        llm_ms = int((time.perf_counter() - llm_start) * 1000)

        latency_ms = int((time.perf_counter() - start) * 1000)
        sources = [c.source for c in contexts]
        logger.info(
            "chat.answer",
            extra=ctx
            | {"latency_ms": latency_ms, "retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "sources": sources},
        )
        return ChatResult(
            answer=answer, sources=sources, latency_ms=latency_ms, retrieval_ms=retrieval_ms, llm_ms=llm_ms
        )

    def _build_prompt(self, question: str, contexts: list[RetrievedContext]) -> str:
        context_block = "\n\n".join(
            f"Source {c.id} ({c.source}, score={c.score}):\n{c.content}" for c in contexts
        )
        template = ChatPromptTemplate.from_template(
            "Use the provided sources to answer the question.\n"
            "Always cite source ids inline like [source:ID]. If unsure, say you do not know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )
        prompt = template.format(context=context_block, question=question)
        return prompt


def get_chat_service(
    session_factory: sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> ChatService:
    settings = get_settings()
    retrieval = get_retrieval_service(session_factory=session_factory, settings=settings)
    return ChatService(retrieval_service=retrieval, settings=settings)
