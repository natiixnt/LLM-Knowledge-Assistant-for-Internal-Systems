from __future__ import annotations

import asyncio

from app.core.logging import setup_logging
from app.db.session import get_session_factory
from app.services.ingestion import IngestionService


async def main() -> None:
    setup_logging()
    session_factory = get_session_factory()
    ingestion = IngestionService(session_factory=session_factory)
    await ingestion.ingest_texts(
        tenant_id="demo-tenant",
        source="example-md",
        texts=[
            "# Welcome\nThis is an example document chunked into pieces.",
            "Internal policy: use strong passwords and 2FA for all accounts.",
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())
