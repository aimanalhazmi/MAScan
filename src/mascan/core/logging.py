import logging
import sys
from typing import Final

from mascan.core.settings import get_settings

LOG_FORMAT: Final = (
    "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
)
DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

configured = False


def configure_logging() -> None:
    global configured
    if configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Call configure_logging() once at startup."""
    return logging.getLogger(name)
