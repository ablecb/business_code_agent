"""Shared pytest fixtures.

The LLM circuit breaker wraps the process-wide ``get_llm_provider()`` singleton
(factory lru_cache). A test that drives the real provider into consecutive
failures (or a slow/absent LLM endpoint) leaves the circuit OPEN, which then
makes later tests that genuinely need LLM output fail fast with the
heuristic/deterministic fallback (e.g. compose-code pipelines) and pollutes the
whole suite. This autouse fixture resets the breaker after every test so the
suite stays isolated from cross-test state, while production behavior is
unaffected.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_llm_circuit_after_each() -> None:
    yield
    try:
        from app.llm import factory
        from app.llm.circuit_breaker import CircuitBreakerLLMProvider

        provider = factory.get_llm_provider()
        if isinstance(provider, CircuitBreakerLLMProvider):
            provider.reset()
    except Exception:
        pass