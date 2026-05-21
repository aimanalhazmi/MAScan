"""Shared HTTP client wrapper. All API tools should use this for consistency.

Provides: sensible timeouts, simple retry on transient errors.
"""
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mascan.core.logging import get_logger

DEFAULT_TIMEOUT = 10.0
logger = get_logger("tools.http_client")


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    reraise=True,
)
def http_get(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Response:
    """GET with timeouts and retry on transient errors."""
    logger.debug("GET %s params=%s", url, params)
    response = httpx.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response