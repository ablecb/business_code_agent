"""LLM fail-fast circuit breaker tests (P1: desktop-chat-5b263372926e 7.3min 根因)."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.llm.base import LLMProvider
from app.llm.circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreakerLLMProvider,
)


class _ControlledProvider(LLMProvider):
    """Each call either fails (TimeoutError) or succeeds per an explicit plan."""

    name = "controlled"

    def __init__(self) -> None:
        self.calls = 0
        self.fail_plan: list[bool] = []

    def plan_fail(self, n: int) -> None:
        """Next n calls fail."""
        self.fail_plan.extend([True] * n)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self.calls += 1
        should_fail = self.fail_plan.pop(0) if self.fail_plan else False
        if should_fail:
            raise TimeoutError(f"simulated LLM down (call {self.calls})")
        return "ok"

    @property
    def model(self) -> str:
        return "test-model"


async def _chat(breaker: CircuitBreakerLLMProvider) -> str:
    return await breaker.chat([{"role": "user", "content": "hi"}])


def test_opens_after_threshold_and_fails_fast():
    async def run() -> None:
        inner = _ControlledProvider()
        inner.plan_fail(2)
        b = CircuitBreakerLLMProvider(inner, fail_threshold=2, open_secs=60.0)
        assert b.state == CLOSED
        with pytest.raises(TimeoutError):
            await _chat(b)  # failure 1/2
        assert b.state == CLOSED
        with pytest.raises(TimeoutError):
            await _chat(b)  # failure 2/2 → OPEN
        assert b.state == OPEN
        # Circuit OPEN: third call raises immediately WITHOUT touching inner.
        calls_before = inner.calls
        t0 = time.perf_counter()
        with pytest.raises(TimeoutError) as ei:
            await _chat(b)
        assert inner.calls == calls_before  # no network hit
        assert time.perf_counter() - t0 < 0.5
        assert "circuit OPEN" in str(ei.value)

    asyncio.run(run())


def test_half_open_probe_reopens_on_failure_then_closes():
    async def run() -> None:
        inner = _ControlledProvider()
        inner.plan_fail(3)  # 2 to open + 1 probe that still fails
        b = CircuitBreakerLLMProvider(inner, fail_threshold=2, open_secs=0.5)
        with pytest.raises(TimeoutError):
            await _chat(b)
        with pytest.raises(TimeoutError):
            await _chat(b)
        assert b.state == OPEN
        await asyncio.sleep(0.6)  # cooldown elapsed → half-open
        assert b.state == HALF_OPEN
        # Probe goes through (inner fails 3rd time) → re-opens.
        with pytest.raises(TimeoutError):
            await _chat(b)
        assert b.state == OPEN
        await asyncio.sleep(0.6)
        # Probe now succeeds → circuit closed.
        out = await _chat(b)
        assert out == "ok"
        assert b.state == CLOSED

    asyncio.run(run())


def test_success_resets_failure_count():
    async def run() -> None:
        inner = _ControlledProvider()
        b = CircuitBreakerLLMProvider(inner, fail_threshold=2, open_secs=60.0)
        inner.plan_fail(1)
        with pytest.raises(TimeoutError):
            await _chat(b)  # failure 1
        out = await _chat(b)  # success → counter reset
        assert out == "ok"
        inner.plan_fail(1)
        with pytest.raises(TimeoutError):
            await _chat(b)  # failure 1 again (not 2 → still closed)
        assert b.state == CLOSED
        inner.plan_fail(1)
        with pytest.raises(TimeoutError):
            await _chat(b)  # failure 2 → OPEN
        assert b.state == OPEN

    asyncio.run(run())


def test_reset():
    async def run() -> None:
        inner = _ControlledProvider()
        inner.plan_fail(2)
        b = CircuitBreakerLLMProvider(inner, fail_threshold=2, open_secs=600.0)
        with pytest.raises(TimeoutError):
            await _chat(b)
        with pytest.raises(TimeoutError):
            await _chat(b)  # OPEN
        assert b.state == OPEN
        b.reset()
        assert b.state == CLOSED
        assert b.failures == 0
        out = await _chat(b)
        assert out == "ok"

    asyncio.run(run())


def test_readonly_name_model_passthrough():
    b = CircuitBreakerLLMProvider(_ControlledProvider())
    assert b.name == "controlled"
    assert b.model == "test-model"


def test_cancelled_outer_waitfor_counts_as_failure():
    """外层的 asyncio.wait_for(timeout) 取消 inner coroutine 也要记为失败，否则
    意图/目标识别的 12s/15s 外壳超时无法推动熔断打开。"""

    class _SlowProvider(LLMProvider):
        name = "slow"

        async def chat(
            self,
            messages,
            *,
            temperature=None,
            max_tokens=None,
            response_format=None,
        ) -> str:
            await asyncio.sleep(1.0)
            return "late"

    async def run() -> None:
        b = CircuitBreakerLLMProvider(_SlowProvider(), fail_threshold=2, open_secs=60.0)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                b.chat([{"role": "user", "content": "hi"}]), timeout=0.02
            )
        assert b.failures == 1
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                b.chat([{"role": "user", "content": "hi"}]), timeout=0.02
            )
        assert b.state == OPEN

    asyncio.run(run())


def test_factory_wraps_get_llm_provider():
    """get_llm_provider() 返回的必须带熔断语义（问答路径），且 name 透传。"""
    from app.llm import factory
    from app.llm.circuit_breaker import CircuitBreakerLLMProvider

    provider = factory.get_llm_provider()
    assert isinstance(provider, CircuitBreakerLLMProvider)
    assert provider.state in {CLOSED, OPEN, HALF_OPEN}