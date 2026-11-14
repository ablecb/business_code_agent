"""Tests for conversation compaction (DSH-style semantic compression)."""
from __future__ import annotations

import pytest

from app.agent.compaction import (
    compact_messages,
    maybe_compact,
    should_compact,
)
from app.llm.mock import MockLLMProvider


def _msgs(n: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(1, n):
        role = "user" if i % 2 else "assistant"
        out.append({"role": role, "content": f"msg-{i}"})
    return out


def test_should_compact_short_conversation_false():
    assert should_compact(_msgs(4)) is False
    assert should_compact(_msgs(10)) is False  # below default threshold 14


def test_should_compact_long_conversation_true():
    assert should_compact(_msgs(20), threshold=14) is True
    assert should_compact(_msgs(22), threshold=12) is True  # 22 > 2+16 tail


def test_compact_messages_keeps_head_and_tail_replaces_middle():
    # Use 20 messages — with default compaction_keep_tail=16,
    # should_compact needs n > 2+16 = 18, so 20 works.
    msgs = _msgs(20)
    import asyncio

    class _CompactingMock:
        async def chat(self, messages, **kwargs):
            return "此前对话摘要：用户问了几个关于流程的问题。"
        async def embed(self, texts, **kwargs):
            return [[0.0] * 8 for _ in texts]

    async def run():
        return await compact_messages(_CompactingMock(), msgs, keep_tail=6, threshold=14)

    result = asyncio.run(run())
    assert result is not None
    # head: system + first user verbatim
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user" and result[1]["content"] == "msg-1"
    # summary note injected at index 2
    assert result[2]["role"] == "system" and "摘要" in result[2]["content"]
    # tail: last 6 verbatim
    assert result[-1]["content"] == "msg-19"
    assert len(result) == 2 + 1 + 6  # head(2) + summary + tail(6)


def test_compact_short_returns_none():
    import asyncio

    async def run():
        return await compact_messages(MockLLMProvider(), _msgs(4))

    assert asyncio.run(run()) is None


def test_maybe_compact_inplace_when_enabled(monkeypatch):
    import asyncio

    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "compaction_enabled", True)
    monkeypatch.setattr(get_settings(), "compaction_threshold", 14)
    msgs = _msgs(20)

    async def run():
        return await maybe_compact(MockLLMProvider(), msgs)

    changed = asyncio.run(run())
    assert changed is True
    assert len(msgs) < 20
    assert any("摘要" in m.get("content", "") for m in msgs)


def test_maybe_compact_skips_when_disabled(monkeypatch):
    import asyncio

    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "compaction_enabled", False)
    msgs = _msgs(20)
    before = list(msgs)

    async def run():
        return await maybe_compact(MockLLMProvider(), msgs)

    assert asyncio.run(run()) is False
    assert msgs == before
