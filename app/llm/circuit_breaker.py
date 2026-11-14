"""LLM fail-fast circuit breaker for the chat/agent path.

Why: when the chat endpoint is unhealthy (connect timeout / transport error /
HTTP 5xx persisting through internal retries), every logical ``chat()`` can burn
the full per-call budget (``LLM_TOTAL_TIMEOUT``, default 90s) × serial calls in
one turn — e.g. desktop-chat-5b263372926e took ~7.3 min on a dead endpoint and
still degraded the intent/goal routing.

The breaker counts consecutive ``chat()`` failures. Once the threshold is hit it
opens: subsequent ``chat()`` calls raise ``TimeoutError`` immediately without
touching the network, so agent fallbacks (heuristic intent / heuristic goal /
deterministic api_inventory & co.) kick in in milliseconds instead of seconds.
After a cooldown it goes half-open and lets a single probe through with a short
timeout; probe success closes the circuit, failure re-opens it.

Scope: only the provider returned by ``factory.get_llm_provider()`` (即问答路径)
is wrapped. The offline parse/ingest pipeline uses ``build_llm_provider()`` /
``get_parse_llm_provider()`` directly and stays unlimited / unaffected.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, AsyncIterator

from app.llm.base import LLMProvider

logger = logging.getLogger("app.llm.circuit")

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreakerLLMProvider(LLMProvider):
    """Fail-fast wrapper around a chat LLM provider (see module docstring)."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        fail_threshold: int = 2,
        open_secs: float = 30.0,
        probe_timeout: float = 15.0,
    ) -> None:
        self._inner = inner
        self._fail_threshold = max(1, int(fail_threshold))
        self._open_secs = max(0.5, float(open_secs))
        self._probe_timeout = max(1.0, float(probe_timeout))
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.RLock()
        self._name = getattr(inner, "name", "breaker")
        self.model = getattr(inner, "model", "")

    # -- introspection passthrough -------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> str:
        """closed | open | half_open"""
        with self._lock:
            if not self._open_until:
                return CLOSED
            if time.monotonic() >= self._open_until:
                return HALF_OPEN
            return OPEN

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def reset(self) -> None:
        """Close the circuit and clear failure counts (tests / hot reload)."""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    # -- breaker core ---------------------------------------------------------
    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._fail_threshold:
                self._open_until = time.monotonic() + self._open_secs
                logger.warning(
                    "LLM circuit OPEN after %s consecutive failures (open %.0fs) "
                    "provider=%s model=%s — chat() fails fast until probe succeeds",
                    self._failures,
                    self._open_secs,
                    self._name,
                    self.model,
                )
            else:
                logger.warning(
                    "LLM circuit failure %s/%s provider=%s model=%s",
                    self._failures,
                    self._fail_threshold,
                    self._name,
                    self.model,
                )

    def _record_success(self) -> None:
        with self._lock:
            was_open = bool(self._open_until)
            self._failures = 0
            self._open_until = 0.0
            if was_open:
                logger.warning(
                    "LLM circuit CLOSED (probe/success) provider=%s model=%s",
                    self._name,
                    self.model,
                )

    def _guard(self) -> bool:
        """True if call should proceed; raises TimeoutError when circuit is OPEN."""
        st = self.state
        if st == OPEN:
            raise TimeoutError(
                "LLM circuit OPEN — chat() failed fast "
                f"(provider={self._name} model={self.model}); "
                "agent falls back to heuristic/deterministic path"
            )
        return st == HALF_OPEN

    # -- LLMProvider interface ------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        probe = self._guard()
        if probe:
            logger.info(
                "LLM circuit half-open probe provider=%s model=%s (%.0fs cap)",
                self._name,
                self.model,
                self._probe_timeout,
            )
        coro = self._inner.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        try:
            if probe:
                out = await asyncio.wait_for(coro, timeout=self._probe_timeout)
            else:
                out = await coro
        except asyncio.CancelledError:
            # Outer wait_for timeouts cancel this coroutine — count as a failure
            # so a dead endpoint opens the circuit from the caller's timeout too.
            self._record_failure()
            raise
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return out

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self._guard()
        async for chunk in self._inner.chat_stream(
            messages, temperature=temperature, max_tokens=max_tokens
        ):
            yield chunk
        self._record_success()