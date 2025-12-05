from __future__ import annotations

import logging
import sys
from typing import Any


class ContextFormatter(logging.Formatter):
    """Simple key=value formatter suitable for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting only
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in {"msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message", "asctime"}
            and not k.startswith("_")
        }
        extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
        return f"{base} {extra_str}".strip()


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(ContextFormatter(fmt=fmt))
    logging.basicConfig(level=level, handlers=[handler])

    # Placeholder for OpenTelemetry integration.
    # To connect to OTLP exporter, attach appropriate handlers here.


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
