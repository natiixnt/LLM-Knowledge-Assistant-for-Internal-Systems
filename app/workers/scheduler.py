from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from app.services.ingestion import IngestionService

logger = logging.getLogger("app.scheduler")


async def periodic_refresh(ingestion: IngestionService, interval: timedelta) -> None:
    """Simple scheduler that triggers periodic refresh. Intended to be run as a background task."""
    while True:  # pragma: no cover - scheduler loop
        try:
            # Placeholder: load changed sources and re-run ingestion.
            logger.info("scheduler.tick")
            # await ingestion.ingest_texts(...)
        except Exception:  # noqa: BLE001
            logger.exception("scheduler.error")
        await asyncio.sleep(interval.total_seconds())
