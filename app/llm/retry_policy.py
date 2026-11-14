"""Shared LLM HTTP retry policy — borrowed from DeepSeek Harness's llm-retry idea.

Centralizes which status codes are retryable, backoff timing, and Retry-After
header respect so every provider (ollama, openai-compatible, …) applies the
same policy instead of each implementing its own ad-hoc loop.
"""
from __future__ import annotations

import time
from typing import Any

# Transient / proxy-ish failures worth retrying (405 often from MITM/nginx when SSL verify off)
RETRY_STATUS = frozenset({405, 408, 425, 429, 500, 502, 503, 504})
DEFAULT_MAX_RETRIES = 5


def should_retry(status_code: int | None) -> bool:
    return status_code in RETRY_STATUS


def parse_retry_after(header: str | None) -> float | None:
    """Parse an HTTP Retry-After header (seconds or HTTP-date) → seconds, or None."""
    if not header:
        return None
    header = header.strip()
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    try:
        # HTTP-date (RFC 7231) — rare; keep a bounded wait.
        parsed = time.mktime(time.strptime(header, "%a, %d %b %Y %H:%M:%S GMT"))
        return max(0.0, parsed - time.time())
    except ValueError:
        return None


def retry_sleep_sec(status_code: int | None, attempt: int) -> float:
    """Backoff for transient LLM errors; 429 needs longer waits (RPM limits)."""
    if status_code == 429:
        # ~2s, 5s, 12s, 25s, 40s (capped)
        return min(40.0, 1.5 * (2.2**attempt))
    return min(8.0, 0.8 * attempt)


def compute_delay(
    status_code: int | None,
    attempt: int,
    retry_after_header: str | None = None,
    *,
    max_delay_ms: float | None = None,
) -> float:
    """Delay before the next attempt, honoring Retry-After when present.

    Mirrors DSH llm-retry: provider Retry-After wins over the policy backoff,
    bounded by an optional max delay (ms).
    """
    delay = retry_sleep_sec(status_code, attempt)
    ra = parse_retry_after(retry_after_header)
    if ra is not None:
        delay = max(delay, ra)
    if max_delay_ms is not None and max_delay_ms > 0:
        delay = min(delay, max_delay_ms / 1000.0)
    return delay


def log_retry(logger: Any, *, provider: str, model: str, attempt: int, max_retries: int, status: int | None, sleep_s: float, body: str = "") -> None:
    logger.warning(
        "LLM %s status=%s attempt=%s/%s model=%s sleep=%.1fs body=%s — retry",
        provider,
        status,
        attempt,
        max_retries,
        model,
        sleep_s,
        (body or "")[:300],
    )
