from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    url: str = Field(..., description="SQLAlchemy database URL, e.g. postgres+asyncpg://user:pass@host/db")
    echo: bool = Field(False, description="Enable SQLAlchemy echo for debugging")


class OpenAISettings(BaseModel):
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-3.5-turbo", description="Chat model name")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model name")
    timeout_seconds: float = Field(8.0, description="Timeout for upstream LLM calls")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "stage", "prod", "test"] = Field("dev", alias="APP_ENV")
    db: DatabaseSettings
    openai: OpenAISettings
    log_level: str = Field("INFO", description="Logging level")
    embeddings_batch_size: int = Field(64, description="Batch size for embedding creation")
    max_context_chunks: int = Field(8, description="Max number of document chunks to include in context")
    similarity_threshold: float = Field(
        0.2, description="Cosine similarity cutoff; lower means looser matching"
    )

    def require_openai(self) -> None:
        if not self.openai.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    try:
        return AppSettings()  # type: ignore[arg-type]
    except ValidationError as exc:
        missing = ", ".join(err["loc"][0] for err in exc.errors())
        raise RuntimeError(f"Missing or invalid configuration: {missing}") from exc
