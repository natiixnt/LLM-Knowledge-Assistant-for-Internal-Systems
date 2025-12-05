from __future__ import annotations

import asyncio

from app.core.logging import setup_logging
from app.db.base import Base
from app.db.models import *  # noqa: F403,F401
from app.db.session import get_engine


async def main() -> None:
    setup_logging()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
