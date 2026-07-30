from __future__ import annotations

import logging
from time import sleep
from typing import Callable, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def call_with_request_retries(fn: Callable[[], T], *, label: str) -> T:
    """Retry a synchronous OpenAI request on transient failures (network,
    rate limit) with linear backoff, up to settings.openai_request_max_attempts.

    Mirrors the request-level retry loop already used by the preview
    endpoint's LLMClient, so the corrections and custom-work endpoints share
    the same resilience policy instead of drifting independently (preview
    retries request failures, the others previously did not).
    """
    last_error: Exception | None = None
    max_attempts = max(settings.openai_request_max_attempts, 1)
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying", label, attempt, max_attempts, exc
            )
            sleep(0.25 * attempt)
    assert last_error is not None
    raise last_error
