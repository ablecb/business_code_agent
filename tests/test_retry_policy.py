"""Tests for shared LLM retry policy (DSH llm-retry style)."""
from __future__ import annotations

from app.llm.retry_policy import (
    compute_delay,
    parse_retry_after,
    retry_sleep_sec,
    should_retry,
)


def test_should_retry_status_codes():
    assert should_retry(429) is True
    assert should_retry(502) is True
    assert should_retry(503) is True
    assert should_retry(504) is True
    assert should_retry(408) is True
    assert should_retry(400) is False
    assert should_retry(401) is False
    assert should_retry(200) is False
    assert should_retry(None) is False


def test_retry_sleep_backoff():
    assert retry_sleep_sec(429, 1) < retry_sleep_sec(429, 3)
    # 429 backoff is longer than generic backoff
    assert retry_sleep_sec(429, 1) > retry_sleep_sec(500, 1)
    # generic backoff caps
    assert retry_sleep_sec(500, 99) <= 8.0
    # 429 backoff caps
    assert retry_sleep_sec(429, 99) <= 40.0


def test_parse_retry_after_seconds():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(" 0.5 ") == 0.5
    assert parse_retry_after("") is None
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None


def test_compute_delay_respects_retry_after():
    # Retry-After wins over the policy backoff
    delay = compute_delay(429, 1, retry_after_header="10")
    assert delay >= 10.0
    # max_delay_ms bounds it
    delay = compute_delay(429, 1, retry_after_header="100", max_delay_ms=5000)
    assert delay <= 5.0
    # no header → policy backoff
    delay = compute_delay(429, 1)
    assert delay == retry_sleep_sec(429, 1)
