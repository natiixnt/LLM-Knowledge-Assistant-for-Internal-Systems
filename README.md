# LLM Knowledge Assistant Backend

An internal, multi-tenant knowledge assistant built with FastAPI, PostgreSQL, SQLAlchemy 2.x (async), LangChain prompt helpers, and OpenAI. It ingests docs + relational data, performs retrieval with embeddings, and serves a `/chat` endpoint with observability hooks.

## Quickstart

1. Install dependencies (Python 3.11+):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install .[dev]
   ```
2. Set env vars (example):
   ```bash
   export APP_ENV=dev
   export DB__URL="postgres+asyncpg://app:app@localhost:5432/appdb"
   export OPENAI__API_KEY="sk-..."
   ```
3. Initialize DB:
   ```bash
   python -m scripts.init_db
   ```
4. Run API:
   ```bash
   uvicorn app.main:app --reload
   ```

## Docker

```bash
OPENAI_API_KEY=sk-... docker compose up --build
```
Postgres with pgvector runs on `localhost:5432`; API on `localhost:8000`.

## API

- `POST /chat` - `{question, user_id, tenant_id?}` → `{answer, sources, latency_ms}` with enforced upstream timeout.
- `GET /health`, `GET /ready` - health/readiness.
- `POST /admin/ingest/text` - trigger ingestion of raw text/markdown for a tenant.
- `GET /admin/status` - last ingestion timestamp.

## Architecture Notes

- **Config**: `app/core/config.py` via `.env`/env vars; fails fast when required fields missing.
- **Models**: `Document`, `DocumentChunk` (embeddings stored as ARRAY on Postgres, JSON fallback elsewhere) and `CustomerAccount` for structured data context.
- **Retrieval**: cosine similarity in Python with clear hook to replace by pgvector SQL search; filters by `tenant_id`, caps chunks, and mixes in structured account summaries.
- **Chat**: `ChatService` composes retrieval + OpenAI chat with LangChain prompt templating, measures retrieval/LLM latency, and logs structured context.
- **Ingestion**: chunk text, embed, persist; includes helper to load customer accounts.
- **Observability**: structured logging ready for OTEL exporters; metrics hooks/log fields for latency and errors.
- **Background work**: `app/workers/scheduler.py` stub for periodic refresh.

## Tests & CI

```bash
pytest
```
GitHub Actions workflow runs ruff format/lint and pytest on each push/PR.

## Scripts

- `python -m scripts.init_db` - create tables.
- `python -m scripts.run_etl_example` - example ingestion of sample text.

## Future Enhancements

- Swap Python cosine scoring with pgvector similarity search.
- Add migrations (Alembic) and richer ETL sources (PDF/HTML parsers, S3/Blob connectors).
- Wire OTLP exporter + metrics backend (Prometheus/OTel Collector).
