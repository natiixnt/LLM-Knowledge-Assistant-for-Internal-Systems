from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI

from app.core.config import AppSettings


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingClient:
    def __init__(self, settings: AppSettings):
        settings.require_openai()
        self._client = AsyncOpenAI(api_key=settings.openai.api_key, timeout=settings.openai.timeout_seconds)
        self._model = settings.openai.embedding_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
